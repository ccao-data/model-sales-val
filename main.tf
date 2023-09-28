terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.16"
    }
  }

  required_version = ">= 1.5.7"

  backend "s3" {
    bucket               = "ccao-terraform-state-us-east-1"
    key                  = "workspaces/default/terraform.tfstate"
    region               = "us-east-1"
    workspace_key_prefix = "workspaces"
  }
}

provider "aws" {
  region = "us-east-1"
}

locals {
  s3_prefix           = "script/sales-val${terraform.workspace == "prod" ? "" : "-${terraform.workspace}"}"
  s3_warehouse_bucket = "s3://ccao-data-warehouse-us-east-1${terraform.workspace == "prod" ? "" : "/sale-dev/${terraform.workspace}"}"
  glue_job_name       = "sales_val_flagging${terraform.workspace == "prod" ? "" : "_${terraform.workspace}"}"
}

variable "iam_role_arn" {
  type        = string
  description = "IAM role ARN to use to run the job"
  sensitive   = true
  nullable    = false
}

variable "commit_sha" {
  type        = string
  description = <<EOF
    "Hash of the current git commit, to use for versioning the model. "
    "Generate with `git rev-parse --short HEAD`"
  EOF
  nullable    = false
}

variable "glue_job_s3_bucket" {
  type        = string
  description = "S3 bucket name where the script and its modules should be stored"
  default     = "ccao-glue-assets-us-east-1"
  nullable    = false
}

resource "aws_s3_object" "sales_val_flagging" {
  bucket = var.glue_job_s3_bucket
  key    = "${local.s3_prefix}/sales_val_flagging.py"
  source = "${path.module}/glue/sales_val_flagging.py"
  etag   = filemd5("${path.module}/glue/sales_val_flagging.py")
}

resource "aws_s3_object" "flagging_script" {
  bucket = var.glue_job_s3_bucket
  key    = "${local.s3_prefix}/flagging.py"
  source = "${path.module}/glue/flagging_script_glue/flagging.py"
}

resource "aws_glue_job" "sales_val_flagging_glue_job" {
  name            = local.glue_job_name
  role_arn        = var.iam_role_arn
  max_retries     = 0
  max_capacity    = "1.0"
  glue_version    = "3.0"
  execution_class = "STANDARD"

  command {
    name            = "pythonshell"
    script_location = aws_s3_object.sales_val_flagging.id
    python_version  = "3.9"
  }

  default_arguments = {
    # TODO: Perhaps we need to manage these buckets with Terraform
    # so that we can delete them?
    "--s3_glue_bucket"            = "s3://${var.glue_job_s3_bucket}"
    "--s3_prefix"                 = "${local.s3_prefix}/"
    "--aws_s3_warehouse_bucket"   = local.s3_warehouse_bucket
    "--enable-job-insights"       = false
    "--region_name"               = "us-east-1"
    "--job-language"              = "python"
    "--TempDir"                   = "${var.glue_job_s3_bucket}/temporary/"
    "--s3_staging_dir"            = "s3://ccao-athena-results-us-east-1"
    "--stat_groups"               = "rolling_window,township_code,class"
    "--iso_forest"                = "meta_sale_price,sv_price_per_sqft,sv_days_since_last_transaction,sv_cgdr,sv_sale_dup_counts"
    "--rolling_window_num"        = 12
    "--time_frame_start"          = "2023-01-01"
    "--dev_bounds"                = "2,3"
    "--additional-python-modules" = "Random-Word==1.0.11,boto3==1.28.12"
    "--commit_sha"                = var.commit_sha
  }
}

import {
  to = aws_glue_crawler.ccao_data_warehouse_sale_crawler
  id = "ccao-data-warehouse-sale-crawler"
}
