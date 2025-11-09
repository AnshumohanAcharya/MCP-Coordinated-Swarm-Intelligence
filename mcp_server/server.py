"""Model Context Protocol server for swarm coordination."""

import asyncio
import websockets
import json
import time
import signal
import sys
from typing import Dict, Set, Optional, Any
from loguru import logger

from config.mcp_config import MCPConfig, ContextMessage
from .context_manager import ContextManager
from .message_protocol import MessageProtocol


class MCPServer:
    """Model Context Protocol server for coordinating UAV swarm intelligence."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.context_manager = ContextManager(config)
        self.message_protocol = MessageProtocol()
        # Use Any instead of deprecated WebSocketServerProtocol
        self.clients: Set[Any] = set()
        self.running = False
        
        # Register message handlers
        self._register_handlers()
        
        # Setup logging
        logger.add("logs/mcp_server.log", rotation="1 day", retention="7 days")
    
    def _register_handlers(self):
        """Register message handlers for different message types."""
        self.message_protocol.register_handler("update", self._handle_update)
        self.message_protocol.register_handler("query", self._handle_query)
        self.message_protocol.register_handler("heartbeat", self._handle_heartbeat)
        self.message_protocol.register_handler("register", self._handle_register)
    
    async def _handle_update(self, client_id: str, message: ContextMessage) -> Optional[str]:
        """Handle context update from client."""
        # Use sender_id (UAV ID) as the key, not websocket client_id
        uav_id = message.sender_id
        logger.info(f"Received update from {uav_id} (context_type: {message.context_type})")
        
        # Update context manager with new data using UAV ID
        self.context_manager.update_client_data(uav_id, message)
        
        # Broadcast aggregated context to all clients
        aggregated_context = self.context_manager.aggregate_context()
        context_message = ContextMessage(
            message_type="context_broadcast",
            sender_id="server",
            timestamp=time.time(),
            context_type="aggregated_context",
            data=aggregated_context.to_dict(),
            priority=2
        )
        
        await self.message_protocol.broadcast_to_clients(context_message, exclude_client=client_id)
        
        # Send acknowledgment
        ack_message = ContextMessage(
            message_type="acknowledgment",
            sender_id="server",
            timestamp=time.time(),
            context_type=message.context_type,
            data={"status": "updated", "client_id": client_id},
            priority=1
        )
        
        return self.message_protocol.serialize_message(ack_message)
    
    async def _handle_query(self, client_id: str, message: ContextMessage) -> str:
        """Handle context query from client."""
        logger.info(f"Received query from {client_id}: {message.context_type}")
        
        # Get context relevant to the client
        context = self.context_manager.get_context_for_client(client_id)
        
        if context:
            response_data = {
                "context": context.to_dict(),
                "query_type": message.context_type,
                "timestamp": time.time()
            }
        else:
            response_data = {
                "error": "No context available",
                "query_type": message.context_type,
                "timestamp": time.time()
            }
        
        response_message = ContextMessage(
            message_type="query_response",
            sender_id="server",
            timestamp=time.time(),
            context_type=message.context_type,
            data=response_data,
            priority=2
        )
        
        return self.message_protocol.serialize_message(response_message)
    
    async def _handle_heartbeat(self, client_id: str, message: ContextMessage) -> str:
        """Handle heartbeat from client."""
        logger.debug(f"Received heartbeat from {client_id}")
        
        # Update client heartbeat
        self.message_protocol.update_client_heartbeat(client_id)
        
        # Return heartbeat response
        return self.message_protocol.create_heartbeat_response()
    
    async def _handle_register(self, client_id: str, message: ContextMessage) -> str:
        """Handle client registration."""
        logger.info(f"Client {client_id} registered")
        
        # Update client info without overwriting websocket
        client_info = message.data.get("client_info", {})
        if client_id in self.message_protocol.registered_clients:
            # Update info but keep existing websocket
            self.message_protocol.registered_clients[client_id]["info"] = client_info
        else:
            # New registration - websocket should already be set by register_client
            # But if not, we can't register without a websocket
            logger.warning(f"Client {client_id} not found in registered clients during register message")
        
        # Send registration confirmation
        response_message = ContextMessage(
            message_type="registration_confirmed",
            sender_id="server",
            timestamp=time.time(),
            context_type="registration",
            data={"client_id": client_id, "status": "registered"},
            priority=1
        )
        
        return self.message_protocol.serialize_message(response_message)
    
    async def register_client(self, websocket: Any, client_id: str):
        """Register a new client connection."""
        self.clients.add(websocket)
        self.message_protocol.register_client(client_id, websocket)
        logger.info(f"Client {client_id} connected. Total clients: {len(self.clients)}")
    
    async def unregister_client(self, websocket: Any, client_id: str):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        self.message_protocol.unregister_client(client_id)
        logger.info(f"Client {client_id} disconnected. Total clients: {len(self.clients)}")
    
    async def handle_client(self, websocket: Any, path: str = None):
        """Handle individual client connection."""
        client_id = f"client_{int(time.time() * 1000)}"  # Simple ID generation
        
        try:
            await self.register_client(websocket, client_id)
            
            async for message in websocket:
                try:
                    # Handle incoming message
                    response = await self.message_protocol.handle_message(client_id, message)
                    
                    if response:
                        try:
                            await websocket.send(response)
                        except websockets.exceptions.ConnectionClosed:
                            logger.debug(f"Connection closed while sending response to {client_id}")
                            break
                        except Exception as e:
                            logger.warning(f"Error sending response to {client_id}: {e}")
                            # Don't break - continue processing other messages
                        
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}", exc_info=True)
                    try:
                        error_response = self.message_protocol.create_error_response(str(e))
                        await websocket.send(error_response)
                    except (websockets.exceptions.ConnectionClosed, Exception):
                        # Connection already closed or can't send error response
                        logger.debug(f"Could not send error response to {client_id}")
                        break
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} connection closed")
        except Exception as e:
            logger.error(f"Error in client handler for {client_id}: {e}")
        finally:
            await self.unregister_client(websocket, client_id)
    
    async def periodic_context_update(self):
        """Periodically update and broadcast context."""
        while self.running:
            try:
                # Clean up old data
                self.context_manager.cleanup_old_data(self.config.context_retention_time)
                
                # Get context summary
                summary = self.context_manager.get_context_summary()
                logger.debug(f"Context summary: {summary}")
                
                # Broadcast periodic context update
                if self.clients:
                    aggregated_context = self.context_manager.aggregate_context()
                    context_message = ContextMessage(
                        message_type="periodic_update",
                        sender_id="server",
                        timestamp=time.time(),
                        context_type="aggregated_context",
                        data=aggregated_context.to_dict(),
                        priority=1
                    )
                    
                    await self.message_protocol.broadcast_to_clients(context_message)
                
                await asyncio.sleep(1.0 / self.config.context_update_frequency)
                
            except Exception as e:
                logger.error(f"Error in periodic context update: {e}")
                await asyncio.sleep(1.0)
    
    async def start_server(self):
        """Start the MCP server."""
        logger.info(f"Starting MCP server on {self.config.host}:{self.config.port}")
        
        self.running = True
        
        # Start periodic context update task
        context_task = asyncio.create_task(self.periodic_context_update())
        
        try:
            # Start WebSocket server
            async with websockets.serve(
                self.handle_client,
                self.config.host,
                self.config.port,
                max_size=self.config.max_message_size,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.message_timeout
            ) as server:
                logger.info("MCP server started successfully")
                logger.info(f"Server listening on ws://{self.config.host}:{self.config.port}")
                await asyncio.Future()  # Run forever
                
        except OSError as e:
            if "address already in use" in str(e).lower() or e.errno == 48:
                logger.error(f"Port {self.config.port} is already in use!")
                logger.error("Another MCP server instance may be running.")
                logger.error("Solution: Kill the existing process or use a different port.")
                logger.error(f"To find and kill: lsof -ti:{self.config.port} | xargs kill -9")
            else:
                logger.error(f"Error starting MCP server: {e}")
        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
        finally:
            self.running = False
            context_task.cancel()
    
    def stop_server(self):
        """Stop the MCP server."""
        logger.info("Stopping MCP server")
        self.running = False


def main():
    """Main function to run the MCP server."""
    # Load configuration
    config = MCPConfig()
    
    # Create and start server
    server = MCPServer(config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        server.stop_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Run server
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
