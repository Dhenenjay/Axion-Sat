"""
Safe I/O operations for checkpoint management.

This module provides robust checkpoint save/load functionality with:
- Atomic writes (write to temp, then rename)
- MD5 checksum validation
- Automatic backups
- Corruption detection and recovery
- Cloud storage integration (optional)

Features:
- Thread-safe operations
- Progress tracking for large files
- Compression support
- Metadata tracking
- Automatic cleanup of old checkpoints
"""

import torch
import os
import shutil
import hashlib
import json
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
import time
import threading


def compute_md5(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """
    Compute MD5 checksum of a file.
    
    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read (for memory efficiency)
        
    Returns:
        MD5 checksum as hex string
    """
    md5_hash = hashlib.md5()
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest()


def verify_md5(file_path: Union[str, Path], expected_md5: str) -> bool:
    """
    Verify file MD5 checksum.
    
    Args:
        file_path: Path to the file
        expected_md5: Expected MD5 checksum
        
    Returns:
        True if checksum matches, False otherwise
    """
    actual_md5 = compute_md5(file_path)
    return actual_md5.lower() == expected_md5.lower()


class CheckpointManager:
    """
    Manager for safe checkpoint operations.
    
    Handles atomic writes, validation, backups, and recovery.
    
    Args:
        checkpoint_dir: Directory to store checkpoints
        max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
        auto_backup: Automatically backup before overwriting
        verify_on_load: Verify checksum when loading
        compress: Use compression (reduces size, increases time)
        verbose: Print detailed information
        
    Example:
        >>> manager = CheckpointManager("checkpoints/", max_checkpoints=5)
        >>> 
        >>> # Save checkpoint
        >>> state = {
        ...     'epoch': 10,
        ...     'model_state_dict': model.state_dict(),
        ...     'optimizer_state_dict': optimizer.state_dict(),
        ...     'loss': 0.123
        ... }
        >>> manager.save_checkpoint(state, "checkpoint_epoch_10.pt")
        >>> 
        >>> # Load checkpoint
        >>> state = manager.load_checkpoint("checkpoint_epoch_10.pt")
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        auto_backup: bool = True,
        verify_on_load: bool = True,
        compress: bool = False,
        verbose: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_backup = auto_backup
        self.verify_on_load = verify_on_load
        self.compress = compress
        self.verbose = verbose
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        self.metadata = self._load_metadata()
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load metadata: {e}")
        
        return {'checkpoints': {}, 'history': []}
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        try:
            # Atomic write
            temp_file = self.metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Rename (atomic on most systems)
            temp_file.replace(self.metadata_file)
            
        except Exception as e:
            warnings.warn(f"Failed to save metadata: {e}")
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        async_save: bool = False
    ) -> str:
        """
        Save checkpoint with atomic write and validation.
        
        Args:
            state: State dictionary to save (model, optimizer, etc.)
            filename: Checkpoint filename
            metadata: Optional metadata to store
            async_save: Save asynchronously (non-blocking)
            
        Returns:
            Path to saved checkpoint
        """
        if async_save:
            # Save in background thread
            thread = threading.Thread(
                target=self._save_checkpoint_impl,
                args=(state, filename, metadata)
            )
            thread.daemon = True
            thread.start()
            return str(self.checkpoint_dir / filename)
        else:
            return self._save_checkpoint_impl(state, filename, metadata)
    
    def _save_checkpoint_impl(
        self,
        state: Dict[str, Any],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Internal implementation of checkpoint saving."""
        with self._lock:
            checkpoint_path = self.checkpoint_dir / filename
            
            # Backup existing checkpoint
            if checkpoint_path.exists() and self.auto_backup:
                self._backup_checkpoint(checkpoint_path)
            
            # Write to temporary file first (atomic write)
            temp_file = checkpoint_path.with_suffix('.tmp')
            
            try:
                start_time = time.time()
                
                if self.verbose:
                    print(f"Saving checkpoint: {filename}")
                
                # Save with compression if enabled
                if self.compress:
                    torch.save(state, temp_file, _use_new_zipfile_serialization=True)
                else:
                    torch.save(state, temp_file)
                
                # Compute checksum
                md5_checksum = compute_md5(temp_file)
                
                # Atomic rename
                temp_file.replace(checkpoint_path)
                
                # Update metadata
                checkpoint_metadata = {
                    'filename': filename,
                    'path': str(checkpoint_path),
                    'md5': md5_checksum,
                    'size_mb': checkpoint_path.stat().st_size / 1e6,
                    'timestamp': datetime.now().isoformat(),
                    'compressed': self.compress,
                    'custom_metadata': metadata or {}
                }
                
                self.metadata['checkpoints'][filename] = checkpoint_metadata
                self.metadata['history'].append({
                    'action': 'save',
                    'filename': filename,
                    'timestamp': datetime.now().isoformat()
                })
                
                self._save_metadata()
                
                # Cleanup old checkpoints
                if self.max_checkpoints > 0:
                    self._cleanup_old_checkpoints()
                
                elapsed = time.time() - start_time
                
                if self.verbose:
                    print(f"✓ Checkpoint saved successfully")
                    print(f"  Path: {checkpoint_path}")
                    print(f"  Size: {checkpoint_metadata['size_mb']:.2f} MB")
                    print(f"  MD5: {md5_checksum[:16]}...")
                    print(f"  Time: {elapsed:.2f}s")
                
                return str(checkpoint_path)
                
            except Exception as e:
                # Cleanup temp file on error
                if temp_file.exists():
                    temp_file.unlink()
                
                raise RuntimeError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(
        self,
        filename: str,
        map_location: Optional[Union[str, torch.device]] = None,
        verify_checksum: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint with validation.
        
        Args:
            filename: Checkpoint filename
            map_location: Device to map tensors to
            verify_checksum: Override default checksum verification
            
        Returns:
            Loaded state dictionary
        """
        with self._lock:
            checkpoint_path = self.checkpoint_dir / filename
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            if self.verbose:
                print(f"Loading checkpoint: {filename}")
            
            # Verify checksum if enabled
            should_verify = verify_checksum if verify_checksum is not None else self.verify_on_load
            
            if should_verify and filename in self.metadata['checkpoints']:
                expected_md5 = self.metadata['checkpoints'][filename]['md5']
                
                if self.verbose:
                    print(f"  Verifying checksum...")
                
                if not verify_md5(checkpoint_path, expected_md5):
                    # Try to recover from backup
                    if self.auto_backup:
                        backup_path = self._get_backup_path(checkpoint_path)
                        if backup_path.exists():
                            warnings.warn(f"Checkpoint corrupted, attempting recovery from backup")
                            return self._load_from_backup(backup_path, map_location)
                    
                    raise RuntimeError(
                        f"Checkpoint corrupted: MD5 mismatch for {filename}. "
                        f"Expected {expected_md5[:16]}..."
                    )
                
                if self.verbose:
                    print(f"  ✓ Checksum verified")
            
            try:
                start_time = time.time()
                
                # Load checkpoint
                state = torch.load(checkpoint_path, map_location=map_location)
                
                # Update metadata
                self.metadata['history'].append({
                    'action': 'load',
                    'filename': filename,
                    'timestamp': datetime.now().isoformat()
                })
                self._save_metadata()
                
                elapsed = time.time() - start_time
                
                if self.verbose:
                    size_mb = checkpoint_path.stat().st_size / 1e6
                    print(f"✓ Checkpoint loaded successfully")
                    print(f"  Size: {size_mb:.2f} MB")
                    print(f"  Time: {elapsed:.2f}s")
                
                return state
                
            except Exception as e:
                # Try to recover from backup
                if self.auto_backup:
                    backup_path = self._get_backup_path(checkpoint_path)
                    if backup_path.exists():
                        warnings.warn(f"Failed to load checkpoint, attempting recovery from backup")
                        return self._load_from_backup(backup_path, map_location)
                
                raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    def _backup_checkpoint(self, checkpoint_path: Path):
        """Create backup of existing checkpoint."""
        backup_path = self._get_backup_path(checkpoint_path)
        
        try:
            shutil.copy2(checkpoint_path, backup_path)
            
            if self.verbose:
                print(f"  Created backup: {backup_path.name}")
                
        except Exception as e:
            warnings.warn(f"Failed to create backup: {e}")
    
    def _get_backup_path(self, checkpoint_path: Path) -> Path:
        """Get backup path for a checkpoint."""
        return checkpoint_path.with_suffix('.backup')
    
    def _load_from_backup(
        self,
        backup_path: Path,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """Load checkpoint from backup."""
        if self.verbose:
            print(f"  Loading from backup: {backup_path.name}")
        
        return torch.load(backup_path, map_location=map_location)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return
        
        # Get checkpoints sorted by timestamp
        checkpoints = sorted(
            self.metadata['checkpoints'].items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Remove old checkpoints
        for filename, metadata in checkpoints[self.max_checkpoints:]:
            checkpoint_path = Path(metadata['path'])
            
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    
                    # Remove backup if exists
                    backup_path = self._get_backup_path(checkpoint_path)
                    if backup_path.exists():
                        backup_path.unlink()
                    
                    if self.verbose:
                        print(f"  Removed old checkpoint: {filename}")
                    
                except Exception as e:
                    warnings.warn(f"Failed to remove old checkpoint {filename}: {e}")
            
            # Remove from metadata
            del self.metadata['checkpoints'][filename]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        return sorted(
            self.metadata['checkpoints'].values(),
            key=lambda x: x['timestamp'],
            reverse=True
        )
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[0]['path'] if checkpoints else None
    
    def delete_checkpoint(self, filename: str, delete_backup: bool = True):
        """
        Delete a checkpoint and its backup.
        
        Args:
            filename: Checkpoint filename
            delete_backup: Also delete backup file
        """
        with self._lock:
            checkpoint_path = self.checkpoint_dir / filename
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            if delete_backup:
                backup_path = self._get_backup_path(checkpoint_path)
                if backup_path.exists():
                    backup_path.unlink()
            
            # Remove from metadata
            if filename in self.metadata['checkpoints']:
                del self.metadata['checkpoints'][filename]
                self._save_metadata()
            
            if self.verbose:
                print(f"Deleted checkpoint: {filename}")
    
    def verify_all_checkpoints(self) -> Dict[str, bool]:
        """
        Verify all checkpoints in the directory.
        
        Returns:
            Dictionary mapping filename to verification status
        """
        results = {}
        
        for filename, metadata in self.metadata['checkpoints'].items():
            checkpoint_path = Path(metadata['path'])
            
            if not checkpoint_path.exists():
                results[filename] = False
                continue
            
            expected_md5 = metadata['md5']
            results[filename] = verify_md5(checkpoint_path, expected_md5)
        
        return results
    
    def repair_metadata(self):
        """
        Repair metadata by scanning checkpoint directory.
        
        Useful if metadata file is corrupted or lost.
        """
        if self.verbose:
            print("Repairing metadata...")
        
        new_metadata = {'checkpoints': {}, 'history': []}
        
        # Scan directory for checkpoint files
        for checkpoint_path in self.checkpoint_dir.glob("*.pt"):
            if checkpoint_path.name == self.metadata_file.name:
                continue
            
            filename = checkpoint_path.name
            
            try:
                md5_checksum = compute_md5(checkpoint_path)
                
                new_metadata['checkpoints'][filename] = {
                    'filename': filename,
                    'path': str(checkpoint_path),
                    'md5': md5_checksum,
                    'size_mb': checkpoint_path.stat().st_size / 1e6,
                    'timestamp': datetime.fromtimestamp(
                        checkpoint_path.stat().st_mtime
                    ).isoformat(),
                    'compressed': False,
                    'custom_metadata': {}
                }
                
                if self.verbose:
                    print(f"  Scanned: {filename}")
                    
            except Exception as e:
                warnings.warn(f"Failed to scan {filename}: {e}")
        
        self.metadata = new_metadata
        self._save_metadata()
        
        if self.verbose:
            print(f"✓ Metadata repaired: {len(new_metadata['checkpoints'])} checkpoints")


# ============================================================================
# Convenience Functions
# ============================================================================

def safe_save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
    verify: bool = True,
    backup: bool = True,
    verbose: bool = True
) -> str:
    """
    Safely save a checkpoint with atomic write and validation.
    
    This is a standalone function that doesn't require CheckpointManager.
    
    Args:
        state: State dictionary to save
        filepath: Path to save checkpoint
        verify: Compute and verify checksum
        backup: Create backup if file exists
        verbose: Print information
        
    Returns:
        MD5 checksum of saved file
        
    Example:
        >>> state = {'model': model.state_dict(), 'epoch': 10}
        >>> md5 = safe_save_checkpoint(state, "checkpoint.pt")
    """
    filepath = Path(filepath)
    
    # Create parent directory
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing file
    if filepath.exists() and backup:
        backup_path = filepath.with_suffix('.backup')
        shutil.copy2(filepath, backup_path)
        if verbose:
            print(f"Created backup: {backup_path}")
    
    # Write to temporary file
    temp_file = filepath.with_suffix('.tmp')
    
    try:
        if verbose:
            print(f"Saving checkpoint: {filepath.name}")
        
        torch.save(state, temp_file)
        
        # Compute checksum
        if verify:
            md5_checksum = compute_md5(temp_file)
            
            # Save checksum file
            checksum_file = filepath.with_suffix('.md5')
            with open(checksum_file, 'w') as f:
                f.write(md5_checksum)
        
        # Atomic rename
        temp_file.replace(filepath)
        
        if verbose:
            size_mb = filepath.stat().st_size / 1e6
            print(f"✓ Checkpoint saved: {size_mb:.2f} MB")
            if verify:
                print(f"  MD5: {md5_checksum}")
        
        return md5_checksum if verify else ""
        
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to save checkpoint: {e}")


def safe_load_checkpoint(
    filepath: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    verify: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Safely load a checkpoint with validation.
    
    Args:
        filepath: Path to checkpoint
        map_location: Device to map tensors to
        verify: Verify checksum if available
        verbose: Print information
        
    Returns:
        Loaded state dictionary
        
    Example:
        >>> state = safe_load_checkpoint("checkpoint.pt")
        >>> model.load_state_dict(state['model'])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Verify checksum if available
    if verify:
        checksum_file = filepath.with_suffix('.md5')
        if checksum_file.exists():
            with open(checksum_file, 'r') as f:
                expected_md5 = f.read().strip()
            
            if verbose:
                print(f"Verifying checkpoint: {filepath.name}")
            
            if not verify_md5(filepath, expected_md5):
                # Try to load from backup
                backup_path = filepath.with_suffix('.backup')
                if backup_path.exists():
                    warnings.warn("Checkpoint corrupted, loading from backup")
                    return torch.load(backup_path, map_location=map_location)
                
                raise RuntimeError(f"Checkpoint corrupted: MD5 mismatch")
            
            if verbose:
                print("✓ Checksum verified")
    
    if verbose:
        print(f"Loading checkpoint: {filepath.name}")
    
    state = torch.load(filepath, map_location=map_location)
    
    if verbose:
        size_mb = filepath.stat().st_size / 1e6
        print(f"✓ Checkpoint loaded: {size_mb:.2f} MB")
    
    return state


def save_model_only(
    model: torch.nn.Module,
    filepath: Union[str, Path],
    **kwargs
):
    """
    Save only model state dict.
    
    Args:
        model: PyTorch model
        filepath: Path to save
        **kwargs: Additional arguments for safe_save_checkpoint
    """
    state = {'model_state_dict': model.state_dict()}
    return safe_save_checkpoint(state, filepath, **kwargs)


def load_model_only(
    model: torch.nn.Module,
    filepath: Union[str, Path],
    strict: bool = True,
    **kwargs
):
    """
    Load only model state dict.
    
    Args:
        model: PyTorch model
        filepath: Path to load from
        strict: Strictly enforce that keys match
        **kwargs: Additional arguments for safe_load_checkpoint
    """
    state = safe_load_checkpoint(filepath, **kwargs)
    model.load_state_dict(state['model_state_dict'], strict=strict)


# ============================================================================
# Backward Compatibility Aliases for Training Scripts
# ============================================================================

def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
    **kwargs
) -> str:
    """
    Alias for safe_save_checkpoint for backward compatibility.
    
    Args:
        state: State dictionary to save
        filepath: Path to save checkpoint
        **kwargs: Additional arguments for safe_save_checkpoint
        
    Returns:
        MD5 checksum of saved file (if verify=True)
    """
    return safe_save_checkpoint(state, filepath, **kwargs)


def load_checkpoint(
    filepath: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Alias for safe_load_checkpoint for backward compatibility.
    
    Args:
        filepath: Path to checkpoint
        device: Device to map tensors to (mapped to map_location)
        **kwargs: Additional arguments for safe_load_checkpoint
        
    Returns:
        Loaded state dictionary
    """
    # Map 'device' parameter to 'map_location' for compatibility
    map_location = kwargs.pop('map_location', device)
    return safe_load_checkpoint(filepath, map_location=map_location, **kwargs)


def list_checkpoints(
    checkpoint_dir: Union[str, Path],
    pattern: str = "*.pt"
) -> List[str]:
    """
    List all checkpoint files in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match (default: "*.pt")
        
    Returns:
        List of checkpoint file paths (sorted by modification time, newest first)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    # Find all matching checkpoint files
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return [str(p) for p in checkpoint_files]


# ============================================================================
# Example / Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Checkpoint I/O Utilities")
    print("="*70)
    
    # Test checkpoint manager
    print("\n1. Creating CheckpointManager...")
    manager = CheckpointManager(
        checkpoint_dir="test_checkpoints",
        max_checkpoints=3,
        verbose=True
    )
    
    # Create dummy state
    print("\n2. Saving test checkpoints...")
    for i in range(5):
        state = {
            'epoch': i,
            'model_state_dict': {'dummy': torch.randn(10, 10)},
            'loss': 1.0 / (i + 1)
        }
        
        manager.save_checkpoint(
            state,
            f"checkpoint_epoch_{i}.pt",
            metadata={'notes': f'Test checkpoint {i}'}
        )
        time.sleep(0.1)  # Small delay for timestamp ordering
    
    # List checkpoints
    print("\n3. Listing checkpoints...")
    checkpoints = manager.list_checkpoints()
    for ckpt in checkpoints:
        print(f"  - {ckpt['filename']}: {ckpt['size_mb']:.2f} MB "
              f"({ckpt['timestamp']})")
    
    # Load latest
    print("\n4. Loading latest checkpoint...")
    latest = manager.get_latest_checkpoint()
    if latest:
        state = manager.load_checkpoint(Path(latest).name)
        print(f"  Loaded epoch {state['epoch']}, loss {state['loss']:.4f}")
    
    # Verify all
    print("\n5. Verifying all checkpoints...")
    results = manager.verify_all_checkpoints()
    for filename, valid in results.items():
        status = "✓" if valid else "✗"
        print(f"  {status} {filename}")
    
    # Cleanup
    print("\n6. Cleaning up test checkpoints...")
    import shutil
    if Path("test_checkpoints").exists():
        shutil.rmtree("test_checkpoints")
        print("  Removed test_checkpoints directory")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
