

import pytest
from pydantic import ValidationError

from src.api.schemas.call import (
    CallEvalRequest,
    CallEvalResponse,
    CallLabels,
    ModelMetadata
)


class TestCallEvalRequest:
    """Tests for call evaluation request schema."""
    
    def test_valid_request(self, sample_transcript):
        """Valid request should be accepted."""
        request = CallEvalRequest(
            call_id="CALL001",
            transcript=sample_transcript,
            duration_seconds=180
        )
        
        assert request.call_id == "CALL001"
        assert len(request.transcript) > 0
        assert request.duration_seconds == 180
    
    def test_minimal_request(self):
        """Request with only required fields."""
        request = CallEvalRequest(
            call_id="CALL001",
            transcript="Agent: Hello\nCustomer: Hi"
        )
        
        assert request.call_id == "CALL001"
        assert request.lead_id is None
        assert request.duration_seconds is None
    
    def test_transcript_too_short(self):
        """Short transcript should fail validation."""
        with pytest.raises(ValidationError):
            CallEvalRequest(
                call_id="CALL001",
                transcript="Hi"  
            )
    
    def test_missing_call_id(self):
        """Missing call_id should fail."""
        with pytest.raises(ValidationError):
            CallEvalRequest(
                transcript="Agent: Hello there"
            )


class TestCallLabels:
    """Tests for call labels structure."""
    
    def test_valid_labels(self):
        """Valid labels should be accepted."""
        labels = CallLabels(
            rapport_building=0.8,
            need_discovery=0.7,
            closing_attempt=0.9,
            compliance_risk=0.1
        )
        
        assert labels.rapport_building == 0.8
        assert labels.compliance_risk == 0.1
    
    def test_labels_boundary_values(self):
        """Boundary values should work."""
        labels = CallLabels(
            rapport_building=0.0,
            need_discovery=1.0,
            closing_attempt=0.5,
            compliance_risk=0.0
        )
        
        assert labels.rapport_building == 0.0
        assert labels.need_discovery == 1.0
    
    def test_labels_out_of_range(self):
        """Values outside 0-1 should fail."""
        with pytest.raises(ValidationError):
            CallLabels(
                rapport_building=1.5,  
                need_discovery=0.7,
                closing_attempt=0.9,
                compliance_risk=0.1
            )


class TestModelMetadata:
    """Tests for model metadata structure."""
    
    def test_valid_metadata(self):
        """Valid metadata should be accepted."""
        metadata = ModelMetadata(
            model_name="llama3.2:3b",
            latency_ms=450,
            input_tokens=500,
            output_tokens=100
        )
        
        assert metadata.model_name == "llama3.2:3b"
        assert metadata.latency_ms == 450
    
    def test_optional_tokens(self):
        """Token counts are optional."""
        metadata = ModelMetadata(
            model_name="test-model",
            latency_ms=100
        )
        
        assert metadata.input_tokens is None
        assert metadata.output_tokens is None


class TestCallEvalResponse:
    """Tests for call evaluation response structure."""
    
    def test_valid_response(self):
        """Valid response structure."""
        response = CallEvalResponse(
            call_id="CALL001",
            quality_score=0.75,
            labels=CallLabels(
                rapport_building=0.8,
                need_discovery=0.7,
                closing_attempt=0.8,
                compliance_risk=0.1
            ),
            summary="Agent handled the call professionally.",
            next_actions=["Schedule site visit", "Send brochure"],
            model_metadata=ModelMetadata(
                model_name="llama3.2:3b",
                latency_ms=450
            ),
            is_good_call=True,
            key_points=["Price discussed", "Visit scheduled"]
        )
        
        assert response.call_id == "CALL001"
        assert response.quality_score == 0.75
        assert response.is_good_call is True
        assert len(response.next_actions) == 2
    
    def test_response_required_fields(self):
        """Response must have all required fields."""
        with pytest.raises(ValidationError):
            CallEvalResponse(
                call_id="CALL001",
                quality_score=0.75
                
            )
    
    def test_quality_score_bounds(self):
        """Quality score must be 0-1."""
        with pytest.raises(ValidationError):
            CallEvalResponse(
                call_id="CALL001",
                quality_score=1.5,  
                labels=CallLabels(
                    rapport_building=0.8,
                    need_discovery=0.7,
                    closing_attempt=0.8,
                    compliance_risk=0.1
                ),
                summary="Test",
                next_actions=[],
                model_metadata=ModelMetadata(
                    model_name="test",
                    latency_ms=100
                ),
                is_good_call=True
            )
    
    def test_response_serialization(self):
        """Response should serialize to dict."""
        response = CallEvalResponse(
            call_id="CALL001",
            quality_score=0.75,
            labels=CallLabels(
                rapport_building=0.8,
                need_discovery=0.7,
                closing_attempt=0.8,
                compliance_risk=0.1
            ),
            summary="Test summary",
            next_actions=["Action 1"],
            model_metadata=ModelMetadata(
                model_name="test-model",
                latency_ms=100
            ),
            is_good_call=True,
            key_points=[]
        )
        
        data = response.model_dump()
        
        assert "call_id" in data
        assert "labels" in data
        assert "model_metadata" in data
        assert isinstance(data["labels"], dict)
