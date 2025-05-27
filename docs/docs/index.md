# Credit Risk Analysis documentation!

## Description

The main goal of this project is to deploy a service capable of predicting the credit scored of people based on financial transactional information.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://s3://anyoneai-datasets/credit-data-2010//data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://s3://anyoneai-datasets/credit-data-2010//data/` to `data/`.


