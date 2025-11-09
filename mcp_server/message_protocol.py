"""Message protocol for MCP communication."""

import json
import asyncio
import websockets
from typing import Dict, Any, Optional, Callable
from dataclasses import asdict
import time

from config.mcp_config import ContextMessage, SwarmContext


class MessageProtocol:
    """Handles message serialization, deserialization, and routing for MCP."""
    
    def __init__(self):
        self.message_handlers: Dict[str, Callable] = {}
        self.registered_clients: Dict[str, Dict[str, Any]] = {}
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type] = handler
    
    def register_client(self, client_id: str, websocket, client_info: Dict[str, Any] = None):
        """Register a new client connection."""
        self.registered_clients[client_id] = {
            "websocket": websocket,
            "last_heartbeat": time.time(),
            "info": client_info or {}
        }
    
    def unregister_client(self, client_id: str):
        """Unregister a client connection."""
        if client_id in self.registered_clients:
            del self.registered_clients[client_id]
    
    def update_client_heartbeat(self, client_id: str):
        """Update client heartbeat timestamp."""
        if client_id in self.registered_clients:
            self.registered_clients[client_id]["last_heartbeat"] = time.time()
    
    def get_active_clients(self) -> Dict[str, Dict[str, Any]]:
        """Get all active clients (with recent heartbeats)."""
        current_time = time.time()
        active_clients = {}
        
        for client_id, client_data in self.registered_clients.items():
            if current_time - client_data["last_heartbeat"] < 30.0:  # 30 second timeout
                active_clients[client_id] = client_data
        
        return active_clients
    
    def serialize_message(self, message: ContextMessage) -> str:
        """Serialize a ContextMessage to JSON string."""
        return json.dumps(message.to_dict())
    
    def deserialize_message(self, data: str) -> Optional[ContextMessage]:
        """Deserialize a JSON string to ContextMessage."""
        try:
            message_dict = json.loads(data)
            return ContextMessage.from_dict(message_dict)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error deserializing message: {e}")
            return None
    
    def serialize_context(self, context: SwarmContext) -> str:
        """Serialize a SwarmContext to JSON string."""
        return json.dumps(context.to_dict())
    
    def deserialize_context(self, data: str) -> Optional[SwarmContext]:
        """Deserialize a JSON string to SwarmContext."""
        try:
            context_dict = json.loads(data)
            return SwarmContext.from_dict(context_dict)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error deserializing context: {e}")
            return None
    
    async def handle_message(self, client_id: str, message_data: str) -> Optional[str]:
        """Handle incoming message from client."""
        message = self.deserialize_message(message_data)
        if not message:
            return None
        
        # Update client heartbeat
        self.update_client_heartbeat(client_id)
        
        # Route message to appropriate handler
        if message.message_type in self.message_handlers:
            try:
                response = await self.message_handlers[message.message_type](
                    client_id, message
                )
                return response
            except Exception as e:
                print(f"Error handling message {message.message_type}: {e}")
                return self.create_error_response(str(e))
        
        return None
    
    def create_error_response(self, error_message: str) -> str:
        """Create an error response message."""
        error_msg = ContextMessage(
            message_type="error",
            sender_id="server",
            timestamp=time.time(),
            context_type="error",
            data={"error": error_message},
            priority=4
        )
        return self.serialize_message(error_msg)
    
    def create_heartbeat_response(self) -> str:
        """Create a heartbeat response message."""
        heartbeat_msg = ContextMessage(
            message_type="heartbeat_response",
            sender_id="server",
            timestamp=time.time(),
            context_type="heartbeat",
            data={"status": "alive"},
            priority=1
        )
        return self.serialize_message(heartbeat_msg)
    
    async def broadcast_to_clients(self, message: ContextMessage, exclude_client: str = None):
        """Broadcast a message to all active clients."""
        active_clients = self.get_active_clients()
        message_data = self.serialize_message(message)
        
        tasks = []
        for client_id, client_data in active_clients.items():
            if client_id != exclude_client:
                tasks.append(self._send_to_client(client_id, message_data))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_to_client(self, client_id: str, message: ContextMessage):
        """Send a message to a specific client."""
        if client_id in self.registered_clients:
            message_data = self.serialize_message(message)
            await self._send_to_client(client_id, message_data)
    
    async def _send_to_client(self, client_id: str, message_data: str):
        """Internal method to send data to a client."""
        try:
            if client_id not in self.registered_clients:
                return  # Client not registered
            
            client_data = self.registered_clients[client_id]
            websocket = client_data.get("websocket")
            
            if websocket is None:
                # Websocket not available, skip sending
                return
            
            # Check if websocket is closed
            if hasattr(websocket, 'closed') and websocket.closed:
                self.unregister_client(client_id)
                return
            
            await websocket.send(message_data)
        except websockets.exceptions.ConnectionClosed:
            # Connection closed - will be cleaned up by handle_client
            pass
        except Exception as e:
            # Log error but don't unregister here - let handle_client do it
            # This prevents race conditions
            pass
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific client."""
        return self.registered_clients.get(client_id, {}).get("info")
    
    def get_all_client_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active clients."""
        active_clients = self.get_active_clients()
        return {client_id: client_data["info"] for client_id, client_data in active_clients.items()}
