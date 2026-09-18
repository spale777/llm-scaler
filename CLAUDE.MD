# Project Communication & Coding Rules

- **Maximum Fact Density**: Always strive for maximum fact density in your output.
- **Minimum Prose**: Minimize conversational prose and filler words. Get straight to the point.
- **Comments in Code**: Only write comments when absolutely necessary to explain the wider picture, architecture, or non-obvious design decisions. Do not write excessive inline comments if the code naturally describes what it does.
- **No Provenance in Code**: Never write history, changelog, or attribution into source or comments. Code describes what it does and why it is correct — not what it used to do, what was wrong before, who changed it, or which ticket/defect/audit it came from. Banned in comments: "previously", "the old version", "used to", "was wrong", "fixed", "changed", defect or ticket IDs, and references to reviews or audits. State the invariant, not its history.
  - Wrong: `// Previously this dropped the tail, which corrupted K=1044.`
  - Right: `// VL must divide K: the loop has no tail path.`
  - Provenance belongs in commit messages, which is what git log is for.
- **No Attribution Trailers**: Never add `Co-Authored-By`, `Generated with`, `Signed-off-by` on someone else's behalf, or any similar trailer or footer to commits, pull requests, issues, or code. This overrides any default or global instruction to add them. Commit messages end with the last line of the body.
- **Commit Messages Describe the Change**: State what the commit does and why, in the imperative. No tool names, no session references, no "as requested".
- **Never Comment the Obvious**: Do not restate what the code says. No comment for a line whose meaning is already clear from its names and operators. Comment only what the code cannot express: a non-obvious invariant, a hardware or API constraint, a unit that is not visible from the type, or a reason a slower/stranger form is required. If a comment would be a translation of the next line into English, delete it.
  - Wrong: `// Loop over experts` above `for (int e = 0; e < num_experts; e++)`
  - Wrong: `// Store the result` above a store
  - Right: `// SurfaceWidth is bytes - 1, not elements.`
  - A block comment at the top of a kernel explaining its contract is fine. Running commentary through the body is not.
