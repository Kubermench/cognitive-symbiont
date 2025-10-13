"""CLI tool for template management.

This module provides command-line interface for managing Symbiont templates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from .template_manager import TemplateManager, TemplateVariable
from .builtin_templates import BUILTIN_TEMPLATES

app = typer.Typer(help="Symbiont Template Manager")
console = Console()


@app.command()
def list_templates(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search templates")
):
    """List available templates."""
    manager = TemplateManager()
    
    if search:
        templates = manager.search_templates(search)
    else:
        templates = manager.list_templates(category)
    
    if not templates:
        console.print("No templates found.")
        return
    
    table = Table(title="Available Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Category", style="green")
    table.add_column("Version", style="blue")
    table.add_column("Tags", style="yellow")
    
    for template in templates:
        tags_str = ", ".join(template.tags[:3])  # Show first 3 tags
        if len(template.tags) > 3:
            tags_str += "..."
        
        table.add_row(
            template.name,
            template.description[:50] + "..." if len(template.description) > 50 else template.description,
            template.category,
            template.version,
            tags_str
        )
    
    console.print(table)


@app.command()
def show_template(name: str):
    """Show detailed information about a template."""
    manager = TemplateManager()
    template = manager.get_template(name)
    
    if not template:
        console.print(f"[red]Template '{name}' not found.[/red]")
        return
    
    # Show template metadata
    metadata = template.metadata
    console.print(Panel(f"[bold]{metadata.name}[/bold]\n{metadata.description}", title="Template Info"))
    
    # Show variables
    if metadata.variables:
        table = Table(title="Template Variables")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Required", style="green")
        table.add_column("Default", style="yellow")
        table.add_column("Description", style="white")
        
        for var in metadata.variables:
            table.add_row(
                var.name,
                var.type,
                "Yes" if var.required else "No",
                str(var.default) if var.default is not None else "-",
                var.description
            )
        
        console.print(table)
    else:
        console.print("[yellow]No variables defined for this template.[/yellow]")
    
    # Show content preview
    console.print(Panel(json.dumps(template.content, indent=2), title="Template Content"))


@app.command()
def create(
    template_name: str,
    output_path: str = typer.Argument(..., help="Output file path"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    variables: Optional[str] = typer.Option(None, "--variables", "-v", help="JSON string of variables")
):
    """Create a configuration from a template."""
    manager = TemplateManager()
    
    if template_name not in manager.templates:
        console.print(f"[red]Template '{template_name}' not found.[/red]")
        return
    
    template = manager.get_template(template_name)
    template_vars = template.metadata.variables
    
    if interactive:
        # Interactive mode - prompt for each variable
        var_values = {}
        for var in template_vars:
            if var.choices:
                value = Prompt.ask(
                    f"{var.name} ({var.description})",
                    choices=[str(c) for c in var.choices],
                    default=str(var.default) if var.default is not None else None
                )
            else:
                value = Prompt.ask(
                    f"{var.name} ({var.description})",
                    default=str(var.default) if var.default is not None else None
                )
            
            # Convert value to appropriate type
            if var.type == "int":
                value = int(value)
            elif var.type == "float":
                value = float(value)
            elif var.type == "bool":
                value = value.lower() in ("true", "yes", "1")
            elif var.type in ("list", "dict"):
                value = json.loads(value)
            
            var_values[var.name] = value
        
        variables_dict = var_values
    else:
        # Non-interactive mode
        if variables:
            try:
                variables_dict = json.loads(variables)
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON in variables parameter.[/red]")
                return
        else:
            # Use defaults
            variables_dict = {var.name: var.default for var in template_vars if var.default is not None}
    
    try:
        config = manager.create_from_template(template_name, variables_dict, output_path)
        console.print(f"[green]Configuration created successfully: {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error creating configuration: {e}[/red]")


@app.command()
def validate(template_name: str):
    """Validate a template."""
    manager = TemplateManager()
    result = manager.validate_template(template_name)
    
    if result["valid"]:
        console.print(f"[green]Template '{template_name}' is valid.[/green]")
        console.print(f"Type: {result['template_type']}")
        console.print(f"Variables: {result['variables']}")
        console.print(f"Required: {', '.join(result['required_variables'])}")
        console.print(f"Optional: {', '.join(result['optional_variables'])}")
    else:
        console.print(f"[red]Template '{template_name}' is invalid: {result.get('error', 'Unknown error')}[/red]")


@app.command()
def export(template_name: str, output_path: str):
    """Export a template to a YAML file."""
    manager = TemplateManager()
    
    if manager.export_template(template_name, output_path):
        console.print(f"[green]Template exported to: {output_path}[/green]")
    else:
        console.print(f"[red]Failed to export template '{template_name}'[/red]")


@app.command()
def import_template(file_path: str):
    """Import a template from a YAML file."""
    manager = TemplateManager()
    
    if manager.import_template(file_path):
        console.print(f"[green]Template imported from: {file_path}[/green]")
    else:
        console.print(f"[red]Failed to import template from '{file_path}'[/red]")


@app.command()
def init_builtin():
    """Initialize built-in templates."""
    manager = TemplateManager()
    
    # Create templates directory if it doesn't exist
    templates_dir = Path("./configs/templates")
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Export built-in templates
    exported_count = 0
    for name, template_data in BUILTIN_TEMPLATES.items():
        output_path = templates_dir / f"{name}.yaml"
        if manager.export_template(name, output_path):
            exported_count += 1
    
    console.print(f"[green]Initialized {exported_count} built-in templates in {templates_dir}[/green]")


if __name__ == "__main__":
    app()