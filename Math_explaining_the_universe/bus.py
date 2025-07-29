"""
bus.py - Global Event Bus System
═══════════════════════════════════

Async message broker with back-pressure control for inter-domain communication.
Implements publish/subscribe pattern for fractal branch coordination.
Handles domain routing, message queuing, and resource management.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Callable, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3

class EventType(Enum):
    """Event types for the system."""
    DOMAIN_SPAWN = "domain_spawn"
    BRANCH_CREATED = "branch_created"
    VIRTUAL_MODULE = "virtual_module"
    RESOURCE_WARNING = "resource_warning"
    SYSTEM_SHUTDOWN = "system_shutdown"
    USER_REQUEST = "user_request"
    INTERNAL_SIGNAL = "internal_signal"

@dataclass
class Message:
    """Event bus message."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.INTERNAL_SIGNAL
    source: str = ""
    target: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "source": self.source,
            "target": self.target,
            "data": self.data,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

@dataclass
class Subscription:
    """Event subscription."""
    domain: str
    handler: Callable[[Message], Any]
    event_types: Set[EventType] = field(default_factory=set)
    active: bool = True
    created_at: float = field(default_factory=time.time)
    message_count: int = 0

class EventBus:
    """Global async event bus for inter-domain communication."""
    
    def __init__(self, max_queue_size: int = 10000, max_processing_time: float = 1.0):
        """Initialize the event bus."""
        self.max_queue_size = max_queue_size
        self.max_processing_time = max_processing_time
        
        # Message queues by priority
        self.message_queues = {
            priority: deque() for priority in MessagePriority
        }
        
        # Subscriptions
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.global_subscriptions: List[Subscription] = []
        
        # Processing state
        self.running = False
        self.processing_task = None
        self.stats = {
            "messages_sent": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_expired": 0,
            "queue_overflows": 0,
            "processing_errors": 0
        }
        
        # Back-pressure control
        self.back_pressure_threshold = max_queue_size * 0.8
        self.back_pressure_active = False
        
        # Locks
        self.queue_lock = asyncio.Lock()
        self.subscription_lock = asyncio.Lock()
        
        logger.info("✅ EventBus initialized")

    async def start(self) -> None:
        """Start the event bus processing."""
        if self.running:
            return
            
        self.running = True
        self.processing_task = asyncio.create_task(self._process_messages())
        logger.info("🚀 EventBus started")

    async def stop(self) -> None:
        """Stop the event bus processing."""
        if not self.running:
            return
            
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
                
        logger.info("🛑 EventBus stopped")

    async def subscribe(self, domain: str, handler: Callable[[Message], Any], 
                       event_types: Optional[Set[EventType]] = None) -> str:
        """Subscribe to events for a domain."""
        async with self.subscription_lock:
            subscription = Subscription(
                domain=domain,
                handler=handler,
                event_types=event_types or set()
            )
            
            if domain == "*":
                self.global_subscriptions.append(subscription)
            else:
                self.subscriptions[domain].append(subscription)
                
            logger.info(f"✅ Subscribed {domain} to event bus")
            return subscription

    async def unsubscribe(self, domain: str, subscription: Subscription = None) -> bool:
        """Unsubscribe from events."""
        async with self.subscription_lock:
            if domain == "*":
                if subscription in self.global_subscriptions:
                    self.global_subscriptions.remove(subscription)
                    return True
            else:
                if domain in self.subscriptions:
                    if subscription:
                        if subscription in self.subscriptions[domain]:
                            self.subscriptions[domain].remove(subscription)
                            return True
                    else:
                        # Remove all subscriptions for domain
                        del self.subscriptions[domain]
                        return True
            return False

    async def publish(self, message: Message) -> bool:
        """Publish a message to the event bus."""
        if not self.running:
            logger.warning("⚠️  EventBus not running, message discarded")
            return False
            
        # Check for back-pressure
        if self._check_back_pressure():
            logger.warning(f"⚠️  Back-pressure active, discarding message: {message.id}")
            self.stats["queue_overflows"] += 1
            return False
            
        # Check if message is expired
        if message.is_expired():
            logger.debug(f"⏰ Message expired: {message.id}")
            self.stats["messages_expired"] += 1
            return False
            
        async with self.queue_lock:
            # Add to appropriate priority queue
            queue = self.message_queues[message.priority]
            queue.append(message)
            
        self.stats["messages_sent"] += 1
        logger.debug(f"📤 Published message: {message.id} (priority: {message.priority.name})")
        return True

    async def send_message(self, event_type: EventType, source: str, target: str = "", 
                          data: Dict[str, Any] = None, priority: MessagePriority = MessagePriority.NORMAL,
                          expires_in: Optional[float] = None) -> bool:
        """Send a message (convenience method)."""
        message = Message(
            event_type=event_type,
            source=source,
            target=target,
            data=data or {},
            priority=priority,
            expires_at=time.time() + expires_in if expires_in else None
        )
        return await self.publish(message)

    def _check_back_pressure(self) -> bool:
        """Check if back-pressure should be applied."""
        total_messages = sum(len(queue) for queue in self.message_queues.values())
        
        if total_messages > self.back_pressure_threshold:
            if not self.back_pressure_active:
                self.back_pressure_active = True
                logger.warning(f"⚠️  Back-pressure activated: {total_messages} messages queued")
            return True
        else:
            if self.back_pressure_active:
                self.back_pressure_active = False
                logger.info("✅ Back-pressure deactivated")
            return False

    async def _process_messages(self) -> None:
        """Main message processing loop."""
        logger.info("🔄 EventBus message processing started")
        
        while self.running:
            try:
                message = await self._get_next_message()
                if message:
                    await self._handle_message(message)
                else:
                    # No messages, brief pause
                    await asyncio.sleep(0.01)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in message processing: {e}")
                self.stats["processing_errors"] += 1
                await asyncio.sleep(0.1)

    async def _get_next_message(self) -> Optional[Message]:
        """Get the next message from priority queues."""
        async with self.queue_lock:
            # Process in priority order
            for priority in MessagePriority:
                queue = self.message_queues[priority]
                if queue:
                    return queue.popleft()
            return None

    async def _handle_message(self, message: Message) -> None:
        """Handle a single message."""
        start_time = time.time()
        
        try:
            # Check if message is expired
            if message.is_expired():
                logger.debug(f"⏰ Message expired during processing: {message.id}")
                self.stats["messages_expired"] += 1
                return
                
            # Find target subscriptions
            target_subscriptions = []
            
            # Global subscriptions
            target_subscriptions.extend(self.global_subscriptions)
            
            # Domain-specific subscriptions
            if message.target:
                if message.target in self.subscriptions:
                    target_subscriptions.extend(self.subscriptions[message.target])
            else:
                # Broadcast to all domains
                for domain_subs in self.subscriptions.values():
                    target_subscriptions.extend(domain_subs)
                    
            # Filter by event type
            filtered_subscriptions = []
            for sub in target_subscriptions:
                if not sub.active:
                    continue
                if not sub.event_types or message.event_type in sub.event_types:
                    filtered_subscriptions.append(sub)
                    
            # Process subscriptions
            if filtered_subscriptions:
                await self._process_subscriptions(message, filtered_subscriptions)
            else:
                logger.debug(f"📭 No subscribers for message: {message.id}")
                
            self.stats["messages_processed"] += 1
            
            # Check processing time
            processing_time = time.time() - start_time
            if processing_time > self.max_processing_time:
                logger.warning(f"⏰ Slow message processing: {processing_time:.2f}s for {message.id}")
                
        except Exception as e:
            logger.error(f"❌ Error handling message {message.id}: {e}")
            self.stats["messages_failed"] += 1
            
            # Retry logic
            if message.retry_count < message.max_retries:
                message.retry_count += 1
                await asyncio.sleep(0.1 * message.retry_count)  # Exponential backoff
                async with self.queue_lock:
                    self.message_queues[message.priority].appendleft(message)
                logger.info(f"🔄 Retrying message: {message.id} (attempt {message.retry_count})")

    async def _process_subscriptions(self, message: Message, subscriptions: List[Subscription]) -> None:
        """Process message for all subscriptions."""
        tasks = []
        
        for subscription in subscriptions:
            try:
                # Create task for each handler
                if asyncio.iscoroutinefunction(subscription.handler):
                    task = asyncio.create_task(subscription.handler(message))
                else:
                    # Wrap sync handler in async
                    task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(
                            None, subscription.handler, message
                        )
                    )
                tasks.append((task, subscription))
                
            except Exception as e:
                logger.error(f"❌ Error creating task for subscription {subscription.domain}: {e}")
                
        # Wait for all handlers with timeout
        if tasks:
            try:
                done, pending = await asyncio.wait(
                    [task for task, _ in tasks],
                    timeout=self.max_processing_time,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    logger.warning(f"⏰ Cancelled slow handler task")
                    
                # Update subscription stats
                for task, subscription in tasks:
                    if task in done:
                        subscription.message_count += 1
                        try:
                            await task  # Get any exceptions
                        except Exception as e:
                            logger.error(f"❌ Handler error in {subscription.domain}: {e}")
                            
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Message processing timeout: {message.id}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        queue_sizes = {
            priority.name: len(queue) 
            for priority, queue in self.message_queues.items()
        }
        
        subscription_counts = {
            domain: len(subs) 
            for domain, subs in self.subscriptions.items()
        }
        
        return {
            **self.stats,
            "running": self.running,
            "back_pressure_active": self.back_pressure_active,
            "queue_sizes": queue_sizes,
            "total_queued": sum(queue_sizes.values()),
            "subscription_counts": subscription_counts,
            "global_subscriptions": len(self.global_subscriptions)
        }

    async def process_events(self) -> None:
        """Process pending events (for external control)."""
        if not self.running:
            await self.start()
            
        # Process a batch of messages
        batch_size = 10
        processed = 0
        
        while processed < batch_size:
            message = await self._get_next_message()
            if not message:
                break
                
            await self._handle_message(message)
            processed += 1

    async def shutdown(self) -> None:
        """Shutdown the event bus gracefully."""
        logger.info("🛑 EventBus shutting down...")
        
        await self.stop()
        
        # Clear all queues
        async with self.queue_lock:
            for queue in self.message_queues.values():
                queue.clear()
                
        # Clear subscriptions
        async with self.subscription_lock:
            self.subscriptions.clear()
            self.global_subscriptions.clear()
            
        logger.info("✅ EventBus shutdown complete")

    def __del__(self):
        """Cleanup on destruction."""
        if self.running and self.processing_task:
            self.processing_task.cancel()

# Global event bus instance
_event_bus = None

def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

if __name__ == "__main__":
    # Test the event bus
    async def test_handler(message: Message):
        print(f"📨 Received: {message.event_type.value} from {message.source}")
        
    async def test_event_bus():
        bus = EventBus()
        await bus.start()
        
        # Subscribe test handler
        await bus.subscribe("test_domain", test_handler, {EventType.USER_REQUEST})
        
        # Send test message
        await bus.send_message(
            EventType.USER_REQUEST,
            source="test_sender",
            target="test_domain",
            data={"content": "Hello, World!"}
        )
        
        # Process messages
        await asyncio.sleep(0.1)
        await bus.process_events()
        
        # Show stats
        stats = await bus.get_stats()
        print(f"📊 Event bus stats: {stats}")
        
        await bus.shutdown()
    
    print("🌟 EventBus test")
    asyncio.run(test_event_bus())
