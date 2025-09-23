import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.xaihandler import xAI_Handler, UserIntent  # Import real UserIntent
from src.personality import AgentPersonality
from pydantic import BaseModel

class TestHandler: 
    @pytest.fixture
    def mock_xai_client(self): 
        """Mock xAI client that returns predictable responses."""
        mock_client = MagicMock()
        return mock_client 
    
    @pytest.fixture
    def handler(self, mock_xai_client, monkeypatch):
        """Create handler with mocked client."""
        # Mock SentenceTransformer globally for this fixture
        mock_st_instance = MagicMock()
        monkeypatch.setattr('src.memory.SentenceTransformer', lambda x: mock_st_instance)
        
        with patch('src.xaihandler.Client', return_value=mock_xai_client):
            return xAI_Handler(api_key="fake_key", validate_connection=False)
        
    def test_chat_basic(self, handler, mock_xai_client): 
        """Test basic chat functionality."""
        # Mock xai_sdk.chat functions at module level
        with patch('src.xaihandler.system') as mock_system, \
             patch('src.xaihandler.user') as mock_user, \
             patch('src.xaihandler.assistant') as mock_assistant:
            
            # Mock the messages to be simple dicts
            mock_system.return_value = {"role": "system", "content": "You are a helpful assistant."}
            mock_assistant.return_value = {"role": "assistant", "content": "Recent context"}
            mock_user.return_value = {"role": "user", "content": "Hello, how are you?"}
            
            # Mock the chat creation and response
            mock_chat = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Hello! I'm doing great, thank you!"
            mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            mock_chat.sample.return_value = mock_response
            mock_xai_client.chat.create.return_value = mock_chat
            
            response = handler.chat("Hello, how are you?")
            
            # Verify API was called
            mock_xai_client.chat.create.assert_called_once()
            mock_xai_client.chat.create.return_value.sample.assert_called_once()
            
            # Verify response
            assert "Hello" in response
            assert len(response) > 0

    def test_structured_intent(self, handler): 
        """Test structured intent detection."""
        # Mock chat_completion to return UserIntent-compatible response WITH usage
        mock_chat_completion = MagicMock()
        mock_chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"type": "book_flight", "params": {"destination": "Tokyo"}, "confidence": 0.95}'
                }
            }],
            "usage": {  # Required for logging
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        with patch.object(handler, 'chat_completion', mock_chat_completion):
            intent = handler.structured_intent(
                "I need to book a flight to Tokyo",
                system_prompt="Detect travel intent"
            )

            # Fix: Check for real UserIntent
            assert isinstance(intent, UserIntent)
            assert intent.type == "book_flight"
            assert intent.confidence == 0.95
            assert intent.params == {"destination": "Tokyo"}

    @patch('src.xaihandler.StatefulMemory')
    def test_config_from_env(self, mock_memory): 
        """Test config loading from environment variables."""
        # Mock memory creation
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        
        # Set environment variables explicitly
        with patch.dict('os.environ', {
            'XAI_API_KEY': 'test_key',
            'XAI_MODEL': 'grok-3',
            'XAI_TIMEOUT': '1800'
        }):
            handler = xAI_Handler(api_key=None, validate_connection=False)

            # Verify environment values were used
            assert handler.api_key == 'test_key'
            assert handler.model == 'grok-3'
            assert handler.timeout == 1800

    def test_document_analysis(self, handler, tmp_path, monkeypatch): 
        """Test PDF document analysis with mock file."""
        # Create fake PDF path
        fake_pdf = tmp_path / "test.pdf"
        
        # Mock Path.exists() to return True
        def mock_exists(path):
            return True
        monkeypatch.setattr('pathlib.Path.exists', mock_exists)
        
        # Mock fitz.open() with proper structure
        def mock_fitz_open(path, *args, **kwargs):
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 1
            
            mock_page = MagicMock()
            def mock_get_text(format):
                if format == "text":
                    return "This is a test document about AI."
                elif format == "dict":
                    return {"blocks": [{"type": 0, "bbox": (0, 0, 612, 792), "lines": [{"bbox": (72, 72, 540, 108), "chars": []}]}]}
                return ""
            mock_page.get_text = mock_get_text
            
            mock_doc.load_page.return_value = mock_page
            mock_doc.close = MagicMock()
            return mock_doc
        
        monkeypatch.setattr('fitz.open', mock_fitz_open)
        
        # Mock the client.chat.create.sample() directly (method uses this, not chat_completion)
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"summary": "Test document on AI", "word_count": 6}'  # Real string
        mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cached_prompt_tokens": 0, "reasoning_tokens": 0}  # Real dict
        mock_chat.sample.return_value = mock_response
        handler.client.chat.create.return_value = mock_chat  # Mock the real client from fixture
        
        class SimpleAnalysis(BaseModel):
            summary: str
            word_count: int
        
        result = handler.structured_document_analysis(
            instruction_prompt="Summarize this document.",
            document=fake_pdf,
            output_schema=SimpleAnalysis
        )

        assert isinstance(result, SimpleAnalysis)
        assert "test document" in result.summary.lower()
        assert result.word_count == 6