"""Model Context Protocol server for swarm coordination."""

import asyncio
import json
import signal
import sys
import time
from typing import Dict, Optional, Set

import websockets
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
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
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
        logger.info(f"Received update from {client_id}: {message.context_type}")
        
        # Update context manager with new data
        self.context_manager.update_client_data(client_id, message)
        
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
        
        # Register client with message protocol
        client_info = message.data.get("client_info", {})
        self.message_protocol.register_client(client_id, None, client_info)
        
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
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol, client_id: str):
        """Register a new client connection."""
        self.clients.add(websocket)
        self.message_protocol.register_client(client_id, websocket)
        logger.info(f"Client {client_id} connected. Total clients: {len(self.clients)}")
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol, client_id: str):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        self.message_protocol.unregister_client(client_id)
        logger.info(f"Client {client_id} disconnected. Total clients: {len(self.clients)}")
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str = None):
        """Handle individual client connection."""
        client_id = f"client_{int(time.time() * 1000)}"  # Simple ID generation
        
        try:
            await self.register_client(websocket, client_id)
            
            async for message in websocket:
                try:
                    # Handle incoming message
                    response = await self.message_protocol.handle_message(client_id, message)
                    
                    if response:
                        await websocket.send(response)
                        
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
                    error_response = self.message_protocol.create_error_response(str(e))
                    await websocket.send(error_response)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} connection closed")
        except Exception as e:
            logger.error(f"Error in client handler for {client_id}: {e}")
        finally:
            await self.unregister_client(websocket, client_id)
    
    async def periodic_context_update(self):
        """Periodically update and broadcast context with adaptive frequency."""
        while self.running:
            try:
                # Clean up old data
                self.context_manager.cleanup_old_data(self.config.context_retention_time)
                
                # Get context summary
                summary = self.context_manager.get_context_summary()
                logger.debug(f"Context summary: {summary}")
                
                # Determine adaptive frequency
                # Default: 5Hz (0.2s interval)
                # Critical: 20Hz (0.05s interval)
                current_frequency = 5.0
                
                # Check for critical events in context
                if self.context_manager.has_critical_events():
                    current_frequency = 20.0
                    logger.info("Critical phase detected: Increasing MCP update frequency to 20Hz")
                
                # Broadcast periodic context update
                if self.clients:
                    aggregated_context = self.context_manager.aggregate_context()
                    context_message = ContextMessage(
                        message_type="periodic_update",
                        sender_id="server",
                        timestamp=time.time(),
                        context_type="aggregated_context",
                        data=aggregated_context.to_dict(),
                        priority=1 if current_frequency < 10 else 3
                    )
                    
                    await self.message_protocol.broadcast_to_clients(context_message)
                
                await asyncio.sleep(1.0 / current_frequency)
                
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
            ):
                logger.info("MCP server started successfully")
                await asyncio.Future()  # Run forever
                
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
