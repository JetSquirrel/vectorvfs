<p align="center">
  <img src="docs/source/_static/img/logo_vectorvfs.png" alt="Banner" width="500" />
</p>

# VectorVFS: Your Filesystem as a Vector Database

Documentation is at [https://vectorvfs.readthedocs.io](https://vectorvfs.readthedocs.io/).

VectorVFS is a lightweight Python package that transforms your Linux filesystem into a vector database by leveraging the native VFS (Virtual File System) extended attributes. Rather than maintaining a separate index or external database, VectorVFS stores vector embeddings directly alongside each file—turning your existing directory structure into an efficient and semantically searchable embedding store.

VectorVFS currently uses Meta's Perception Encoders (PE) [[arxiv]](https://arxiv.org/abs/2504.13181) which
includes image/video encoders for vision language understanding, it outperforms InternVL3, Qwen2.5VL
and SigLIP2 for zero-shot image tasks. More models support coming soon.

## Key Features

- **Zero-overhead indexing**  
  Embeddings are stored as extended attributes (xattrs) on each file, eliminating the need for external index files or services.

- **Seamless retrieval**  
  Perform searches across your filesystem, retrieving files by embedding similarity.

- **Flexible embedding support**  
  Plug in any embedding model—from pre-trained transformers to custom feature extractors—and let VectorVFS handle storage and lookup.

- **Lightweight and portable**  
  Built on native Linux VFS functionality, VectorVFS requires no additional daemons, background processes, or databases.
- **Optional AWS S3 Vectors**  
  Keep xattr storage as-is, or sync vectors to [Amazon S3 Vectors](https://aws.amazon.com/s3/features/vectors/) with simple environment switches (local-first, S3-first, or disabled).

## AWS S3 vector storage (optional)

You can mirror or move vector storage to AWS S3 Vectors without changing your workflow. Control behavior via environment variables:

- `VECTORVFS_S3_MODE`: `disabled` (default), `local_primary` (write xattrs, also sync to S3), or `s3_primary` (use S3 first, fall back to local if needed).
- `VECTORVFS_S3_BUCKET`: Name of the S3 vector bucket.
- `VECTORVFS_S3_INDEX`: Target S3 vector index.
- `VECTORVFS_S3_REGION`: AWS region for the S3 Vectors client (falls back to `AWS_REGION`).

When S3 settings are omitted or disabled, VectorVFS continues to store embeddings only in extended attributes.
