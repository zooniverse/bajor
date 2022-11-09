# How are these files used

These files are stored in SCM here but are also stored in the batch blob storage system in the 'training' & 'predictions' containers under the `/code` path prefix. These files are automatically synced to the above locations via a Github Action (see deploy_batch_scipts.yml).

These files are then injected to each Zoobot batch job run at runtime before calling the relevant script.

This setup allows the dev team to iterate quickly by modifying the script artefact on the blob storage system and on the next job run we have new runtime behaviours. Thus we avoid a SCM review / deploy cycle via image building to get code changes into zoobot**.

Long term as the system converges to a known set a scripts we can remove this feature and bake these into the `zoobot-batch` docker image and add CI/CD system to push this image to GHCR on SCM changes.

** We need to rebuild the docker image and push to GHCR for each change and while not impossible during the discovery phase this was too slow.
