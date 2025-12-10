

import pytest
import asyncio
from typing import Generator

import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_lead():
    """Sample lead data for testing."""
    return {
        "lead_id": "TEST001",
        "source": "referral",
        "budget": 15000000,
        "city": "Mumbai",
        "property_type": "3BHK",
        "last_activity_minutes_ago": 30,
        "past_interactions": 5,
        "notes": "Very interested, wants to visit this weekend. Ready to book!",
        "status": "contacted"
    }


@pytest.fixture
def sample_cold_lead():
    """Sample cold lead for testing."""
    return {
        "lead_id": "TEST002",
        "source": "social_media",
        "budget": 2500000,
        "city": "Delhi",
        "property_type": "1BHK",
        "last_activity_minutes_ago": 15000,
        "past_interactions": 0,
        "notes": "Just browsing, not serious. Wrong number provided.",
        "status": "new"
    }


@pytest.fixture
def sample_transcript():
    """Sample call transcript for testing."""
    return """
    Agent: Good morning! Am I speaking with Mr. Sharma?
    Customer: Yes, speaking.
    Agent: Sir, this is Rahul from Premium Properties. You had enquired about the 3BHK apartment in Gurgaon.
    Customer: Yes, I'm interested but the price seems high.
    Agent: I understand your concern, sir. Let me explain the value you're getting. This is a premium location with metro connectivity.
    Customer: What about parking?
    Agent: We provide two covered parking spots included in the price.
    Customer: That sounds reasonable. When can I visit?
    Agent: How about this Saturday at 11 AM? I'll personally show you the property.
    Customer: Yes, that works. Please send me the location.
    Agent: Absolutely, sir. I'll send it right away. Looking forward to meeting you!
    """


@pytest.fixture
def sample_bad_transcript():
    """Sample bad call transcript for testing."""
    return """
    Agent: Hello?
    Customer: Hi, I saw your ad.
    Agent: Which ad? We have many.
    Customer: The one for flats.
    Agent: Budget?
    Customer: 50 lakhs maybe.
    Agent: That's too low. Can't help.
    Customer: Oh, okay then.
    Agent: Bye.
    """
