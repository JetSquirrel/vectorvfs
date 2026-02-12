from dataclasses import dataclass, field
from heapq import heappush, heappushpop
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from rich.console import Console
from PIL import Image

from vectorvfs.config import S3Mode, load_s3_config
from vectorvfs.s3store import S3VectorStore
from vectorvfs.utils import PerfCounter, pillow_image_extensions
from vectorvfs.vfsstore import VFSStore, XAttrFile

console = Console()


@dataclass(order=True)
class PathSimilarity:
    path: Path = field(compare=False)
    similarity: float


@click.group()
def vfs():
    """VectorVFS command line interface."""
    pass


@vfs.command()
@click.option(
    '-n', '--num', 'n', required=True, type=int, default=5,
    help='Number of results to return'
)
@click.argument('query', type=str)
@click.argument(
    'path',
    type=str,
    metavar='PATH')
@click.option('--force-reindex', '-f', is_flag=True, default=False, help="Forces reindexing.")
@click.option('--recursive', '-r', is_flag=True, default=False, help="Recursive search.")
def search(n: int, query: str, path: str, force_reindex: bool,
           recursive: bool) -> None:
    """Search files by similarity."""
    with console.status("", speed=1, spinner="bouncingBall") as status:
        status.update("Loading Perception Encoder model...")

        with PerfCounter() as model_counter:
            from vectorvfs.encoders import PerceptionEncoder
            pe_encoder = PerceptionEncoder()

        console.log(f"Perception Encoder model [bold cyan]{pe_encoder.model_name}[/bold cyan] "
                    f"loaded in [bold cyan]{model_counter.elapsed:.2f}s[/bold cyan].")

        status.update("Encoding search query...")
        with PerfCounter() as query_counter:
            query_features = pe_encoder.encode_text(query)
            query_features = F.normalize(query_features)

        console.log(f"Query encoded in [bold cyan]{query_counter.elapsed:.2f}s[/bold cyan].")
        status.update("Processing files...")

        if path.startswith("s3:"):
            status.update("Querying S3 vectors...")
            s3_config = load_s3_config()
            if s3_config.mode == S3Mode.DISABLED:
                s3_config.mode = S3Mode.S3_PRIMARY
            s3_store = S3VectorStore(s3_config)
            results = s3_store.query(query_features, n)
            if not results:
                console.log("No results found in S3 vectors.")
                return

            console.log(f"\nTop {len(results)} S3 vectors found:")
            for key, score in results[:n]:
                console.log(f"[bold blue]{key}[/bold blue] (Similarity -> {score:.3f})")
            return

        similarity_heap: list[PathSimilarity] = []

        if recursive:
            iter_dir = Path(path).rglob("*")
        else:
            iter_dir = Path(path).iterdir()

        supported_images = pillow_image_extensions()
        for pathfile in iter_dir:
            if not pathfile.is_file():
                continue

            if pathfile.suffix not in supported_images:
                continue

            console.log(f"Processing [bold blue]{pathfile}[/bold blue]")
            xattrfile = XAttrFile(pathfile)
            vfs_store = VFSStore(xattrfile)
            keys = xattrfile.list()
            if "user.vectorvfs" not in keys or force_reindex:
                console.log(f"[bold blue]{pathfile}[/bold blue] not indexed, indexing...")
                try:
                    features = pe_encoder.encode_vision(pathfile)
                except:
                    console.log(f"Failed to index [bold blue]{pathfile}[/bold blue], "
                                "skipping...")
                    continue
                
                features = F.normalize(features)
                features = features.to(torch.float16)
                vfs_store.write_tensor(features)
            else:
                features = vfs_store.read_tensor()

            features = features.to(torch.float32)
            text_probs = features @ query_features.T
            path_similarity = PathSimilarity(pathfile, text_probs.item())

            if len(similarity_heap) < n:
                heappush(similarity_heap, path_similarity)
            else:
                heappushpop(similarity_heap, path_similarity)

        similarity_heap = sorted(similarity_heap, reverse=True)
        console.log(f"\nTop {len(similarity_heap)} files found:")
        for item in similarity_heap[:n]:
            console.log(f"[bold blue]{item.path.name}[/bold blue] "
                        f"(Similarity -> {item.similarity:.3f})")


if __name__ == '__main__':
    vfs()
