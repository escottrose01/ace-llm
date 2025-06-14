"""
CLI formatting utilities for ACE agent.
Provides colored, formatted output for enhanced user experience.
"""

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


class CLIFormatter:
    """Enhanced CLI formatter using Rich library for colored, formatted output."""

    def __init__(self):
        self.console = Console()

    def print_welcome_banner(self):
        """Display the welcome banner with ACE branding."""
        banner = Text()
        banner.append("ACE", style="bold bright_blue")
        banner.append(" agent CLI", style="bold white")

        welcome_panel = Panel(
            banner,
            title="[bold bright_green]Welcome[/bold bright_green]",
            subtitle="Type your query and press Enter. Ctrl+C to exit.",
            border_style="bright_blue",
            padding=(1, 2),
        )
        self.console.print(welcome_panel)

    def print_configuration(self, llm_model: str, embedding_model: str, manifest: str, temperature: float):
        """Display configuration information in a formatted table."""
        config_table = Table(title="Configuration", box=box.ROUNDED, show_header=True, header_style="bold bright_cyan")
        config_table.add_column("Setting", style="bold", width=15)
        config_table.add_column("Value", style="bright_white")

        config_table.add_row("LLM Model", llm_model)
        config_table.add_row("Embedding Model", embedding_model)
        config_table.add_row("Tool Manifest", manifest)
        config_table.add_row("Temperature", str(temperature))

        self.console.print(config_table)

    def print_query_prompt(self) -> str:
        """Print a stylized query prompt and return user input."""
        self.console.print()
        query_text = Text("Query", style="bold bright_yellow")
        query_text.append(" > ", style="dim")
        self.console.print(query_text, end="")
        return input()

    def print_section_header(self, title: str, color: str = "bright_cyan"):
        """Print a section header with consistent styling."""
        self.console.print()
        header_rule = Rule(title, style=f"bold {color}")
        self.console.print(header_rule)

    def print_abstract_tools(self, tools: list[Any]):
        """Display abstract tools in a formatted table."""
        if not tools:
            self.console.print("[dim]No abstract tools generated[/dim]")
            return

        tools_table = Table(title="Abstract Tools", box=box.SIMPLE, show_header=True, header_style="bold bright_cyan")
        tools_table.add_column("Tool Name", style="bold bright_yellow", width=20)
        tools_table.add_column("Description", style="white")

        for tool in tools:
            tools_table.add_row(tool.name, tool.description)

        self.console.print(tools_table)

    def print_plan(self, script: str):
        """Display the generated plan with syntax highlighting."""
        if not script.strip():
            self.console.print("[dim]No plan generated[/dim]")
            return

        plan_syntax = Syntax(script, "python", theme="monokai", line_numbers=True, padding=1)
        plan_panel = Panel(
            plan_syntax,
            title="[bold bright_green]Generated Plan[/bold bright_green]",
            border_style="bright_green",
            padding=(0, 1),
        )
        self.console.print(plan_panel)

    def print_tool_mapping(self, tool_mapping: dict[str, Any]):
        """Display tool mapping in a formatted table."""
        if not tool_mapping:
            no_mapping_text = Text("⚠ No concrete tools could be mapped", style="bold yellow")
            no_mapping_panel = Panel(
                no_mapping_text, title="[yellow]Tool Mapping[/yellow]", border_style="yellow", padding=(1, 2)
            )
            self.console.print(no_mapping_panel)
            return

        mapping_table = Table(title="Tool Mapping", box=box.SIMPLE, show_header=True, header_style="bold bright_cyan")
        mapping_table.add_column("Abstract Tool", style="bold bright_yellow", width=25)
        mapping_table.add_column("Concrete Tool", style="bright_green", width=25)
        mapping_table.add_column("Status", style="bold", width=10)

        for abstract_name, concrete_tool in tool_mapping.items():
            status = "[green]✓ Mapped[/green]"
            mapping_table.add_row(abstract_name, concrete_tool.name, status)

        self.console.print(mapping_table)

    def print_execution_start(self):
        """Display execution start message."""
        self.print_section_header("BEGINNING EXECUTION", "bright_magenta")
        self.console.print("[bold bright_white]Executing plan...[/bold bright_white]")

    def print_execution_complete(self, result: Any):
        """Display execution completion with result."""
        self.print_section_header("EXECUTION COMPLETED", "bright_green")

        if result is not None:
            result_panel = Panel(
                str(result),
                title="[bold bright_green]Result[/bold bright_green]",
                border_style="bright_green",
                padding=(1, 2),
            )
            self.console.print(result_panel)
        else:
            self.console.print("[dim]No result returned[/dim]")

    def print_error(self, error: str):
        """Display error messages in a formatted panel."""
        error_panel = Panel(
            f"[bold red]{error}[/bold red]", title="[bold red]Error[/bold red]", border_style="red", padding=(1, 2)
        )
        self.console.print(error_panel)

    def print_error_with_context(self, error: str, context: str | None = None):
        """Display error messages with additional context information."""
        error_content = f"[bold red]{error}[/bold red]"
        if context:
            error_content += f"\n\n[dim]{context}[/dim]"

        error_panel = Panel(error_content, title="[bold red]Error[/bold red]", border_style="red", padding=(1, 2))
        self.console.print(error_panel)

    def print_success(self, message: str):
        """Display success messages."""
        success_text = Text("✓ ", style="bold green")
        success_text.append(message, style="green")
        self.console.print(success_text)

    def print_warning(self, message: str):
        """Display warning messages."""
        warning_text = Text("⚠ ", style="bold yellow")
        warning_text.append(message, style="yellow")
        self.console.print(warning_text)

    def print_info(self, message: str):
        """Display informational messages."""
        info_text = Text("i ", style="bold blue")
        info_text.append(message, style="blue")
        self.console.print(info_text)

    def print_exit_message(self):
        """Display exit message."""
        self.console.print()
        exit_text = Text("Goodbye! ", style="bold bright_blue")
        exit_text.append("Thank you for using ACE.", style="dim")
        self.console.print(exit_text)

    def create_progress_spinner(self, description: str):
        """Create a progress spinner for long-running operations."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        )

    def print_execution_output(self, message: str):
        """Display formatted execution output from the plan."""
        # Create a nicely formatted panel for the output
        output_panel = Panel(
            message,
            title="[bold bright_cyan]Plan Output[/bold bright_cyan]",
            border_style="bright_cyan",
            padding=(0, 1),
            width=80,
        )
        self.console.print(output_panel)
