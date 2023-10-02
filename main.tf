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
  s3_prefix                = "scripts/sales-val${terraform.workspace == "prod" ? "" : "-${terraform.workspace}"}"
  s3_bucket_data_warehouse = terraform.workspace == "prod" ? "ccao-data-warehouse-us-east-1" : aws_s3_bucket.data_warehouse[0].id
  s3_bucket_glue_assets    = terraform.workspace == "prod" ? "ccao-glue-assets-us-east-1" : aws_s3_bucket.glue_assets[0].id
  glue_job_name            = "sales_val_flagging${terraform.workspace == "prod" ? "" : "_${terraform.workspace}"}"
  glue_crawler_name        = "ccao-data-warehouse-sale-crawler${terraform.workspace == "prod" ? "" : "-${terraform.workspace}"}"
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

resource "aws_s3_bucket" "glue_assets" {
  # Prod buckets are managed outside this config
  count         = terraform.workspace == "prod" ? 0 : 1
  bucket        = "ccao-tmp-glue-assets-${terraform.workspace}-us-east-1"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "glue_assets" {
  count  = terraform.workspace == "prod" ? 0 : 1
  bucket = local.s3_bucket_glue_assets

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "data_warehouse" {
  count         = terraform.workspace == "prod" ? 0 : 1
  bucket        = "ccao-tmp-data-warehouse-${terraform.workspace}-us-east-1"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "data_warehouse" {
  count  = terraform.workspace == "prod" ? 0 : 1
  bucket = local.s3_bucket_data_warehouse

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "sales_val_flagging" {
  bucket = local.s3_bucket_glue_assets
  key    = "${local.s3_prefix}/sales_val_flagging.py"
  source = "${path.module}/glue/sales_val_flagging.py"
  etag   = filemd5("${path.module}/glue/sales_val_flagging.py")
}

resource "aws_s3_object" "flagging_script" {
  bucket = local.s3_bucket_glue_assets
  key    = "${local.s3_prefix}/flagging.py"
  source = "${path.module}/glue/flagging_script_glue/flagging.py"
}

resource "aws_glue_job" "sales_val_flagging" {
  name            = local.glue_job_name
  role_arn        = var.iam_role_arn
  max_retries     = 0
  max_capacity    = "1.0"
  glue_version    = "3.0"
  execution_class = "STANDARD"

  command {
    name            = "pythonshell"
    script_location = "s3://${aws_s3_object.sales_val_flagging.id}"
    python_version  = "3.9"
  }

  default_arguments = {
    "--s3_glue_bucket"            = local.s3_bucket_glue_assets
    "--s3_prefix"                 = "${local.s3_prefix}/"
    "--aws_s3_warehouse_bucket"   = "s3://${local.s3_bucket_data_warehouse}"
    "--enable-job-insights"       = false
    "--region_name"               = "us-east-1"
    "--job-language"              = "python"
    "--TempDir"                   = "s3://${local.s3_bucket_glue_assets}/temporary/"
    "--s3_staging_dir"            = "s3://ccao-athena-results-us-east-1"
    "--stat_groups"               = "rolling_window,township_code,class"
    "--iso_forest"                = "meta_sale_price,sv_price_per_sqft,sv_days_since_last_transaction,sv_cgdr,sv_sale_dup_counts"
    "--rolling_window_num"        = 12
    "--time_frame_start"          = "2023-01-01"
    "--dev_bounds"                = "2,3"
    "--additional-python-modules" = "Random-Word==1.0.11,boto3==1.28.12"
    "--commit_sha"                = var.commit_sha
    "--min_groups_threshold"      = "30"
    "--ptax_sd"                   = "1,1"
  }
}

resource "aws_glue_crawler" "ccao_data_warehouse_sale_crawler" {
  name          = local.glue_crawler_name
  database_name = "sale"
  role          = "ccao-glue-service-role"

  configuration = jsonencode({
    Version = 1,
    Grouping = {
      TableLevelConfiguration = 3
    }
  })

  s3_target {
    path = "s3://${local.s3_bucket_data_warehouse}/sale"
  }

  schema_change_policy {
    delete_behavior = "DELETE_FROM_DATABASE"
    update_behavior = "UPDATE_IN_DATABASE"
  }
}
