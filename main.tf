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
    region  = "us-east-1"
}

variable "iam_role_arn" {
    type        = string
    description = "IAM role ARN to use to run the job"
    sensitive   = true
    nullable    = false
}

variable "glue_job_s3_bucket" {
    type        = string
    description = "S3 bucket name where the script and its modules should be stored"
    default     = "ccao-glue-assets-us-east-1"
    nullable    = false
}

resource "aws_s3_object" "sales_val_flagging_s3_object" {
    bucket = var.glue_job_s3_bucket
    key    = "script/sales-val${terraform.workspace == "prod" ? "" : "-${terraform.workspace}"}/sales_val_flagging.py"
    source = "${path.module}/glue/sales_val_flagging.py"
    etag   = "${filemd5("${path.module}/glue/sales_val_flagging.py")}"
}

resource "aws_glue_job" "sales_val_flagging_glue_job" {
    name            = "sales_val_flagging${terraform.workspace == "prod" ? "" : "_${terraform.workspace}"}"
    role_arn        = var.iam_role_arn
    max_retries     = 0
    max_capacity    = "1.0"
    glue_version    = "3.0"
    execution_class = "STANDARD"

    command {
        name            = "pythonshell"
        script_location = aws_s3_object.sales_val_flagging_s3_object.id
        python_version  = "3.9"
    }

    default_arguments = {
        "--s3_glue_bucket"            = "s3://${var.glue_job_s3_bucket}"
        "--s3_prefix"                 = "script/sales-val${terraform.workspace == "prod" ? "" : "-${terraform.workspace}"}/"
        "--aws_s3_warehouse_bucket"   = "s3://ccao-data-warehouse-us-east-1"
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
    }
}
