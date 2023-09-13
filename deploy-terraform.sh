#!/bin/env bash
# Deploy the sales_val_flagging Glue job as a Terraform plan.
#
# Assumes terraform is installed and that the repo has already been initialized
# for terraform with `terraform init`
#
# Requires one positional env argument, which should be the name of a workspace
# in Terraform remote state (e.g. `prod` or `dev-jecochr`). If you need to use
# a workspace that doesn't exist yet, create one with `terraform workspace new`.
#
# Assumes the IAM_ROLE_ARN environment variables is set.
#
# Example:
#
#   IAM_ROLE_ARN="arn:aws:iam::12345:role/foobar" ./deploy-terraform.sh dev
set -euo pipefail

if [ -z "$1" ]; then
    echo "Missing required positional workspace argument"
    exit 1
fi

WORKSPACE="$1"

terraform workspace select "$WORKSPACE"

terraform apply -auto-approve -var "iam_role_arn=${IAM_ROLE_ARN}"
