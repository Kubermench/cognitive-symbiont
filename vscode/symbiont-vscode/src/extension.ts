import * as vscode from 'vscode';
import { exec } from 'child_process';

export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand('symbiont.proposeTinyRefactor', async () => {
    const term = vscode.window.createOutputChannel('Symbiont');
    term.show(true);
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    term.appendLine('Running: python -m symbiont.cli propose_here');
    exec('python -m symbiont.cli propose_here', { cwd }, async (err, stdout, stderr) => {
      if (err) { term.appendLine('Error: ' + err.message); }
      if (stdout) term.appendLine(stdout);
      if (stderr) term.appendLine(stderr);
      // Try to open latest script
      exec('python -m symbiont.cli latest_artifact --type script', { cwd }, async (e2, out2, err2) => {
        if (!e2 && out2 && out2.trim()) {
          try {
            const p = vscode.Uri.file(require('path').join(cwd || '', out2.trim()));
            const doc = await vscode.workspace.openTextDocument(p);
            await vscode.window.showTextDocument(doc);
            vscode.window.showInformationMessage('Opened latest script. Review and run guarded in UI/CLI.');
          } catch (e) {
            term.appendLine('Open error: ' + (e as Error).message);
          }
        } else {
          vscode.window.showInformationMessage('Symbiont proposal complete. Check Artifacts in data/artifacts.');
        }
      });
    });
  });
  context.subscriptions.push(disposable);

  const runLatest = vscode.commands.registerCommand('symbiont.runLatestScriptSafely', async () => {
    const term = vscode.window.createOutputChannel('Symbiont');
    term.show(true);
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    term.appendLine('Finding latest script...');
    exec('python -m symbiont.cli latest_artifact --type script', { cwd }, async (e2, out2, err2) => {
      if (e2 || !out2.trim()) {
        vscode.window.showWarningMessage('No latest script found. Run a cycle first.');
        if (err2) term.appendLine(err2);
        return;
      }
      const path = out2.trim();
      const pick = await vscode.window.showWarningMessage(`Run script safely? ${path}`, 'Run', 'Cancel');
      if (pick !== 'Run') return;
      term.appendLine('Running guarded: python -m symbiont.cli run_script --yes ' + path);
      exec(`python -m symbiont.cli run_script --yes "${path}"`, { cwd }, (e3, out3, err3) => {
        if (e3) term.appendLine('Error: ' + e3.message);
        if (out3) term.appendLine(out3);
        if (err3) term.appendLine(err3);
        vscode.window.showInformationMessage('Executed latest script. Check exec logs in data/artifacts/logs/.');
      });
    });
  });
  context.subscriptions.push(runLatest);
}

export function deactivate() {}
