# Deploy Experiments to AWS GPU Instance

The scripts in this directory will deploy one of the experiments in
this repository to an AWS GPU instance, specifically, a
[P2 instance with a single Tesla K80](https://aws.amazon.com/ec2/instance-types/p2/).

The experiment should have the following features
- learning pipelines should be persisted/checkpointed to rackspace object storage
- An in-brower REPL should be used, so that the experiment can be monitored

## Usage

Either create a `settings.json` in the deploy directory

```json
{
  "name": "perceptualautoencoder",
  "path": "../perceptual-autoencoder/perceptual_loss.py",
  "amazonec2_access_key": "AWS_ACCESS_KEY",
  "amazonec2_secret_key": "AWS_SECRET",
  "object_storage_region": "DFW",
  "object_storage_username": "RACKSPACE_USERNAME",
  "object_storage_api_key": "RACKSPACE_API_KEY",
  "app_secret": "IN_BROWSER_REPL_PASSWORD"
}
```

Or use the CLI to provide them

```bash
sudo python deploy.py --help
usage: deploy.py [-h] --name NAME --path PATH
                 [--object-storage-region OBJECT_STORAGE_REGION]
                 --object-storage-username OBJECT_STORAGE_USERNAME
                 --object-storage-api-key OBJECT_STORAGE_API_KEY
                 [--app-secret APP_SECRET] --amazonec2-access-key
                 AMAZONEC2_ACCESS_KEY --amazonec2-secret-key
                 AMAZONEC2_SECRET_KEY

optional arguments:
  -h, --help            show this help message and exit

experiment:
  Experiment settings

  --name NAME           the name of the experiment
  --path PATH           absolute or relative path to experimtn python script

object-storage:
  Rackspace object storage settings for model checkpoint storage

  --object-storage-region OBJECT_STORAGE_REGION
                        the rackspace object storage region
  --object-storage-username OBJECT_STORAGE_USERNAME
                        rackspace cloud username
  --object-storage-api-key OBJECT_STORAGE_API_KEY
                        rackspace cloud api key

app:
  In-browser REPL settings

  --app-secret APP_SECRET
                        app password. If not provided, REPL is public

aws:
  AWS EC2 settings

  --amazonec2-access-key AMAZONEC2_ACCESS_KEY
                        aws access key
  --amazonec2-secret-key AMAZONEC2_SECRET_KEY
                        aws secret key
```


## TODO:
- Image the machine after it has been secured, and the nvidia docker runtime
is installed, and start from this image instead
- Provide options for different GPU instances