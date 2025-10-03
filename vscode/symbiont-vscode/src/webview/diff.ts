export function renderDiffHtml(diff: string): string {
  const escaped = diff
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Symbiont Diff Preview</title>
<style>
  body {
    font-family: var(--vscode-editor-font-family, 'Courier New', monospace);
    background: var(--vscode-editor-background, #1e1e1e);
    color: var(--vscode-editor-foreground, #d4d4d4);
    margin: 0;
    padding: 1rem;
    white-space: pre;
  }
  pre {
    margin: 0;
    line-height: 1.4;
  }
</style>
</head>
<body>
<pre>${escaped || 'No diff output.'}</pre>
</body>
</html>`;
}
