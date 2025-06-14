import queue
from unittest.mock import MagicMock, patch

import pytest

from src.ace.execute.orchestrator import PlanOrchestrator


@patch("src.ace.execute.orchestrator.PlanExecutionSandbox")
@patch("src.ace.execute.orchestrator.ToolExecutionSandbox")
def test_plan_orchestrator_lifecycle(mock_tool_sandbox, mock_plan_sandbox):
    plan = MagicMock()
    tools = {"foo": MagicMock()}
    orchestrator = PlanOrchestrator(plan, tools)
    with orchestrator:
        orchestrator.launch()
        orchestrator.plan_execution_sandbox.wait_for_port.assert_called()
        orchestrator.plan_execution_sandbox.launch.assert_called()
        orchestrator.kill()
    # Should be able to call join without error (simulate no exceptions)
    orchestrator.listener_thread = MagicMock()
    orchestrator.listener_thread.join = lambda: None
    orchestrator.exception_queue.empty = lambda: True
    orchestrator.join()


@patch("src.ace.execute.orchestrator.PlanExecutionSandbox")
@patch("src.ace.execute.orchestrator.ToolExecutionSandbox")
def test_plan_orchestrator_error_handling(mock_tool_sandbox, mock_plan_sandbox):
    """Test that errors in sandboxes are properly propagated"""
    # Setup mocks
    plan = MagicMock()
    tools = {"calculator": MagicMock()}

    # Create an orchestrator instance
    orchestrator = PlanOrchestrator(plan, tools)

    # Setup error condition in sandbox
    test_exception = RuntimeError("Sandbox error")
    orchestrator.exception_queue = queue.Queue()
    orchestrator.exception_queue.put(test_exception)

    # Add a mock listener thread so join() can work
    orchestrator.listener_thread = MagicMock()

    # Test that join properly raises the exception
    with pytest.raises(RuntimeError) as exc_info:
        orchestrator.join()

    assert str(exc_info.value) == "Sandbox error"

    # Test error during launch
    orchestrator = PlanOrchestrator(plan, tools)
    orchestrator.plan_execution_sandbox = mock_plan_sandbox.return_value  # Ensure sandbox is instantiated
    orchestrator.plan_execution_sandbox.launch.side_effect = ValueError("Launch failed")

    with pytest.raises(ValueError) as exc_info:
        orchestrator.launch()

    assert "Launch failed" in str(exc_info.value)


@patch("src.ace.execute.orchestrator.PlanExecutionSandbox")
@patch("src.ace.execute.orchestrator.ToolExecutionSandbox")
def test_plan_orchestrator_message_handling(mock_tool_sandbox, mock_plan_sandbox):
    """Test orchestrator message handling functionality"""
    plan = MagicMock()
    tools = {"test_tool": MagicMock()}
    orchestrator = PlanOrchestrator(plan, tools)

    # Test handle_invoke
    tool_args = ["arg1", "arg2"]
    tool_kwargs = {"key": "value"}

    # Mock the tool execution sandbox
    mock_tool_sandbox_instance = MagicMock()
    mock_tool_sandbox.return_value = mock_tool_sandbox_instance

    orchestrator.handle_invoke("test_tool", tool_args, tool_kwargs)

    # Verify tool sandbox was created and launched
    mock_tool_sandbox.assert_called_with(tools["test_tool"])
    mock_tool_sandbox_instance.launch.assert_called_with(tool_args, tool_kwargs)
    assert "test_tool" in orchestrator.tool_use_history


@patch("src.ace.execute.orchestrator.PlanExecutionSandbox")
@patch("src.ace.execute.orchestrator.ToolExecutionSandbox")
def test_plan_orchestrator_tool_execution_failure(mock_tool_sandbox, mock_plan_sandbox):
    """Test orchestrator handling of tool execution failures"""
    plan = MagicMock()
    tools = {"failing_tool": MagicMock()}
    orchestrator = PlanOrchestrator(plan, tools)

    # Setup plan execution sandbox mock
    mock_plan_sandbox = MagicMock()
    orchestrator.plan_execution_sandbox = mock_plan_sandbox

    # Mock tool sandbox to fail on launch
    mock_tool_sandbox_instance = MagicMock()
    mock_tool_sandbox_instance.launch.side_effect = RuntimeError("Tool launch failed")
    mock_tool_sandbox.return_value = mock_tool_sandbox_instance

    # Handle invoke should catch the error and send error message
    orchestrator.handle_invoke("failing_tool", [], {})

    # Verify error was queued and error message sent
    assert not orchestrator.exception_queue.empty()
    error = orchestrator.exception_queue.get()
    assert "Failed to execute tool failing_tool" in str(error)

    # Verify error message was sent to plan sandbox
    mock_plan_sandbox.send.assert_called()
    sent_message = mock_plan_sandbox.send.call_args[0][0]
    assert sent_message.startswith("ERROR:")


def test_plan_orchestrator_print_handling():
    """Test orchestrator print message handling"""
    plan = MagicMock()
    tools = {}

    # Test with no event handler - should fall back to print
    orchestrator = PlanOrchestrator(plan, tools)
    test_message = "Test print message"

    # Mock print to capture output
    with patch("builtins.print") as mock_print:
        orchestrator.handle_print(test_message)
        mock_print.assert_called_with("PRINT:", test_message)

    # Test with event handler
    mock_event_handler = MagicMock()
    orchestrator_with_handler = PlanOrchestrator(plan, tools, event_handler=mock_event_handler)

    orchestrator_with_handler.handle_print(test_message)
    mock_event_handler.on_execution_output.assert_called_once()
    call_args = mock_event_handler.on_execution_output.call_args[0][0]
    assert call_args.message == test_message


def test_plan_orchestrator_terminate_handling():
    """Test orchestrator terminate message handling"""
    plan = MagicMock()
    tools = {}
    orchestrator = PlanOrchestrator(plan, tools)

    # Mock the plan execution sandbox
    mock_plan_sandbox = MagicMock()
    orchestrator.plan_execution_sandbox = mock_plan_sandbox

    # Test handle_terminate
    orchestrator.handle_terminate()

    # Verify sandbox was killed and shutdown event set
    mock_plan_sandbox.kill.assert_called()
    assert orchestrator.shutdown.is_set()


def test_plan_orchestrator_initialization():
    """Test orchestrator proper initialization"""
    plan = MagicMock()
    tools = {"tool1": MagicMock(), "tool2": MagicMock()}

    orchestrator = PlanOrchestrator(plan, tools)

    # Verify initialization
    assert orchestrator.plan == plan
    assert orchestrator.tools == tools
    assert orchestrator.result is None
    assert not orchestrator.shutdown.is_set()
    assert orchestrator.plan_execution_sandbox is None
    assert orchestrator.tool_execution_sandbox is None
    assert len(orchestrator.toolrunner_ids) == 0
    assert orchestrator.exception_queue.empty()
    assert len(orchestrator.tool_use_history) == 0
