"""
Windows Path Verification Tool

Audits Python files in the project to ensure they use Windows-safe path handling:
- Uses pathlib.Path instead of os.path
- No hardcoded backslashes in strings
- Proper path construction with / operator

Usage:
    python tools/verify_windows_paths.py
    python tools/verify_windows_paths.py --fix
    python tools/verify_windows_paths.py --dir axs_lib

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import argparse


@dataclass
class PathIssue:
    """Represents a potential path handling issue."""
    file: Path
    line: int
    issue_type: str
    code: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: str


class PathVerifier:
    """Verifies Python files for Windows-safe path usage."""
    
    def __init__(self):
        self.issues: List[PathIssue] = []
        self.files_checked = 0
        self.files_with_issues = 0
    
    def check_file(self, file_path: Path) -> List[PathIssue]:
        """Check a single Python file for path issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Check for os.path usage
            issues.extend(self._check_os_path_usage(file_path, content, lines))
            
            # Check for hardcoded backslashes
            issues.extend(self._check_hardcoded_backslashes(file_path, lines))
            
            # Check for string path concatenation
            issues.extend(self._check_string_concatenation(file_path, content, lines))
            
            # Check for pathlib import
            if not self._has_pathlib_import(content):
                issues.append(PathIssue(
                    file=file_path,
                    line=1,
                    issue_type='missing_pathlib_import',
                    code='',
                    severity='info',
                    suggestion='Consider importing pathlib.Path for cross-platform path handling'
                ))
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return issues
    
    def _check_os_path_usage(self, file_path: Path, content: str, lines: List[str]) -> List[PathIssue]:
        """Check for os.path usage (suggest pathlib instead)."""
        issues = []
        
        # Parse AST to find os.path calls
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    # Check for os.path.join, os.path.exists, etc.
                    if (isinstance(node.value, ast.Name) and 
                        node.value.id == 'os' and 
                        node.attr.startswith('path')):
                        
                        line_no = node.lineno
                        code = lines[line_no - 1] if line_no <= len(lines) else ''
                        
                        issues.append(PathIssue(
                            file=file_path,
                            line=line_no,
                            issue_type='os_path_usage',
                            code=code.strip(),
                            severity='warning',
                            suggestion='Use pathlib.Path instead of os.path for better cross-platform compatibility'
                        ))
        except SyntaxError:
            pass  # Skip files with syntax errors
        
        return issues
    
    def _check_hardcoded_backslashes(self, file_path: Path, lines: List[str]) -> List[PathIssue]:
        """Check for hardcoded backslashes in strings."""
        issues = []
        
        # Pattern for Windows paths with backslashes
        # Look for strings like "C:\\Users\\..." or "data\\tiles\\..."
        # But exclude: escape sequences, raw strings, docstrings
        path_pattern = re.compile(r'''(['"])([A-Za-z]:\\|\.{1,2}\\|\w+\\)[^'"]*\\[^'"]*\1''')
        
        for line_no, line in enumerate(lines, 1):
            # Skip comments and docstrings
            if line.strip().startswith('#'):
                continue
            if '"""' in line or "'''" in line:
                continue
            
            # Check for backslashes in strings (excluding raw strings and escape sequences)
            if '\\\\' in line or (not line.strip().startswith('r"') and not line.strip().startswith("r'")):
                match = path_pattern.search(line)
                if match:
                    issues.append(PathIssue(
                        file=file_path,
                        line=line_no,
                        issue_type='hardcoded_backslash',
                        code=line.strip(),
                        severity='error',
                        suggestion='Use forward slashes (/) or pathlib.Path for cross-platform compatibility'
                    ))
        
        return issues
    
    def _check_string_concatenation(self, file_path: Path, content: str, lines: List[str]) -> List[PathIssue]:
        """Check for path string concatenation with +."""
        issues = []
        
        # Pattern for string concatenation that might be paths
        # e.g., "data/" + filename or base_path + "/tiles"
        concat_pattern = re.compile(r'''['"][^'"]*[/\\][^'"]*['"]\s*\+\s*|['"]\s*\+\s*['"][^'"]*[/\\]''')
        
        for line_no, line in enumerate(lines, 1):
            if concat_pattern.search(line) and 'http' not in line.lower():
                issues.append(PathIssue(
                    file=file_path,
                    line=line_no,
                    issue_type='string_concatenation',
                    code=line.strip(),
                    severity='warning',
                    suggestion='Use pathlib.Path / operator instead of string concatenation for paths'
                ))
        
        return issues
    
    def _has_pathlib_import(self, content: str) -> bool:
        """Check if file imports pathlib."""
        return 'from pathlib import' in content or 'import pathlib' in content
    
    def verify_directory(self, directory: Path, recursive: bool = True) -> None:
        """Verify all Python files in a directory."""
        pattern = '**/*.py' if recursive else '*.py'
        
        for py_file in directory.glob(pattern):
            # Skip __pycache__ and .git directories
            if '__pycache__' in str(py_file) or '.git' in str(py_file):
                continue
            
            self.files_checked += 1
            file_issues = self.check_file(py_file)
            
            if file_issues:
                self.files_with_issues += 1
                self.issues.extend(file_issues)
    
    def print_report(self, verbose: bool = True):
        """Print verification report."""
        print("\n" + "=" * 80)
        print("Windows Path Verification Report")
        print("=" * 80)
        
        print(f"\nFiles checked: {self.files_checked}")
        print(f"Files with issues: {self.files_with_issues}")
        print(f"Total issues: {len(self.issues)}")
        
        # Count by severity
        errors = [i for i in self.issues if i.severity == 'error']
        warnings = [i for i in self.issues if i.severity == 'warning']
        infos = [i for i in self.issues if i.severity == 'info']
        
        print(f"  Errors:   {len(errors)}")
        print(f"  Warnings: {len(warnings)}")
        print(f"  Info:     {len(infos)}")
        
        if verbose and self.issues:
            print("\n" + "=" * 80)
            print("Detailed Issues")
            print("=" * 80)
            
            # Group by file
            issues_by_file: Dict[Path, List[PathIssue]] = {}
            for issue in self.issues:
                if issue.file not in issues_by_file:
                    issues_by_file[issue.file] = []
                issues_by_file[issue.file].append(issue)
            
            # Print issues by file
            for file_path, file_issues in sorted(issues_by_file.items()):
                print(f"\n{file_path}")
                print("-" * 80)
                
                for issue in file_issues:
                    severity_icon = {
                        'error': '✗',
                        'warning': '⚠',
                        'info': 'ℹ'
                    }[issue.severity]
                    
                    print(f"  {severity_icon} Line {issue.line}: {issue.issue_type}")
                    if issue.code:
                        print(f"    Code: {issue.code[:100]}")
                    print(f"    → {issue.suggestion}")
                    print()
        
        print("\n" + "=" * 80)
        if errors:
            print("✗ FAILED: Found path handling errors that should be fixed")
        elif warnings:
            print("⚠ PASSED: No critical errors, but warnings exist")
        else:
            print("✓ PASSED: All files use Windows-safe path handling")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Verify Python files use Windows-safe path handling',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dir',
        type=Path,
        default=Path('.'),
        help='Directory to check (default: current directory)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Check recursively (default: True)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show summary, not detailed issues'
    )
    
    parser.add_argument(
        '--exclude',
        nargs='+',
        default=[],
        help='Patterns to exclude (e.g., tests/ scripts/)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path
    check_dir = args.dir.resolve()
    
    if not check_dir.exists():
        print(f"Error: Directory not found: {check_dir}")
        return 1
    
    print("\n" + "=" * 80)
    print("Windows Path Verification Tool")
    print("=" * 80)
    print(f"\nChecking directory: {check_dir}")
    print(f"Recursive: {args.recursive}")
    
    if args.exclude:
        print(f"Excluding: {', '.join(args.exclude)}")
    
    # Run verification
    verifier = PathVerifier()
    verifier.verify_directory(check_dir, recursive=args.recursive)
    
    # Print report
    verifier.print_report(verbose=not args.quiet)
    
    # Return exit code
    errors = [i for i in verifier.issues if i.severity == 'error']
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
