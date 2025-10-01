import * as path from 'path';
import * as vscode from 'vscode';
import { exec } from 'child_process';

import { renderDiffHtml } from './webview/diff';

type CliResult = { stdout: string; stderr: string; error?: Error };

function runCli(cwd: string | undefined, command: string): Promise<CliResult> {
  return new Promise((resolve) => {
    exec(command, { cwd }, (error, stdout, stderr) => {
      resolve({ stdout: stdout ?? '', stderr: stderr ?? '', error: error ?? undefined });
    });
  });
}

function appendResult(term: vscode.OutputChannel, result: CliResult) {
  if (result.error) {
    term.appendLine(`Error: ${result.error.message}`);
  }
  if (result.stdout) {
    term.append(result.stdout);
  }
  if (result.stderr) {
    term.append(result.stderr);
  }
}

function shellQuote(value: string): string {
  return `"${value.replace(/"/g, '\\"')}"`;
}

export function activate(context: vscode.ExtensionContext) {
  const proposeCmd = vscode.commands.registerCommand('symbiont.proposeTinyRefactor', async () => {
    const term = vscode.window.createOutputChannel('Symbiont');
    term.show(true);
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    term.appendLine('Running: python -m symbiont.cli propose_here');
    const propose = await runCli(cwd, 'python -m symbiont.cli propose_here');
    appendResult(term, propose);

    const latest = await runCli(cwd, 'python -m symbiont.cli latest_artifact --type script');
    const scriptPath = latest.stdout.trim();
    if (!scriptPath) {
      vscode.window.showInformationMessage('Symbiont proposal complete. Check data/artifacts for outputs.');
      return;
    }
    try {
      const fullPath = path.isAbsolute(scriptPath) ? scriptPath : path.join(cwd ?? '', scriptPath);
      const uri = vscode.Uri.file(fullPath);
      const doc = await vscode.workspace.openTextDocument(uri);
      await vscode.window.showTextDocument(doc);
      vscode.window.showInformationMessage('Opened latest script. Review before executing.');
    } catch (err) {
      term.appendLine('Open error: ' + (err as Error).message);
    }
  });
  context.subscriptions.push(proposeCmd);

  const runLatest = vscode.commands.registerCommand('symbiont.runLatestScriptSafely', async () => {
    const term = vscode.window.createOutputChannel('Symbiont');
    term.show(true);
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    term.appendLine('Finding latest script...');
    const latest = await runCli(cwd, 'python -m symbiont.cli latest_artifact --type script');
    appendResult(term, latest);
    const scriptPath = latest.stdout.trim();
    if (!scriptPath) {
      vscode.window.showWarningMessage('No latest script found. Run a cycle first.');
      return;
    }
    const pick = await vscode.window.showWarningMessage(`Run script safely? ${scriptPath}`, 'Run', 'Cancel');
    if (pick !== 'Run') {
      return;
    }
    const command = `python -m symbiont.cli run_script --yes ${shellQuote(scriptPath)}`;
    term.appendLine('Running guarded: ' + command);
    const result = await runCli(cwd, command);
    appendResult(term, result);
    vscode.window.showInformationMessage('Executed latest script. Check logs under data/artifacts/logs/.');
  });
  context.subscriptions.push(runLatest);

  const showDiff = vscode.commands.registerCommand('symbiont.showProposalDiff', async () => {
    const term = vscode.window.createOutputChannel('Symbiont');
    term.show(true);
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    const latest = await runCli(cwd, 'python -m symbiont.cli latest_artifact --type script');
    appendResult(term, latest);
    const scriptPath = latest.stdout.trim();
    if (!scriptPath) {
      vscode.window.showWarningMessage('No apply script available to diff.');
      return;
    }
    term.appendLine('Generating sandbox diff preview...');
    const diff = await runCli(cwd, `python -m symbiont.cli script_diff ${shellQuote(scriptPath)}`);
    appendResult(term, diff);
    const panel = vscode.window.createWebviewPanel('symbiontDiff', 'Symbiont Proposal Diff', vscode.ViewColumn.Active, { enableScripts: true });
    const content = diff.stdout || diff.stderr || 'No diff output. Script may be a no-op.';
    panel.webview.html = renderDiffHtml(content);
  });
  context.subscriptions.push(showDiff);
}

export function deactivate() {}
