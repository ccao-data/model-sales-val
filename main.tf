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
    key                  = "terraform.tfstate"
    region               = "us-east-1"
    workspace_key_prefix = "model-sales-val/workspaces"
  }
}

provider "aws" {
  region = "us-east-1"
}

locals {
  s3_prefix                = "scripts/sales-val"
  s3_bucket_data_warehouse = terraform.workspace == "prod" ? "ccao-data-warehouse-us-east-1" : aws_s3_bucket.data_warehouse[0].id
  s3_bucket_glue_assets    = terraform.workspace == "prod" ? "ccao-glue-assets-us-east-1" : aws_s3_bucket.glue_assets[0].id
  glue_job_name            = terraform.workspace == "prod" ? "sales_val_flagging" : "ci_${terraform.workspace}_sales_val_flagging"
  glue_crawler_name        = terraform.workspace == "prod" ? "ccao-data-warehouse-sale-crawler" : "ci_${terraform.workspace}_ccao-data-warehouse-sale-crawler"
  glue_table_sale_flag_parameters = {
    CrawlerSchemaDeserializerVersion = "1.0"
    CrawlerSchemaSerializerVersion   = "1.0"
    UPDATED_BY_CRAWLER               = aws_glue_crawler.ccao_data_warehouse_sale_crawler.id
    averageRecordSize                = "17"
    classification                   = "parquet"
    compressionType                  = "none"
    objectCount                      = "1"
    recordCount                      = "69934"
    sizeKey                          = "720294"
    typeOfData                       = "file"
  }
  # Athena databases cannot have hyphens, so replace them with underscores
  # (Note that this is not always true -- notably, dbt-athena is able to
  # create Athena tables with hyphens -- but it's a rule that Terraform
  # enforces, so we follow it here)
  athena_database_name = terraform.workspace == "prod" ? "sale" : "ci_model_sales_val_${replace(terraform.workspace, "-", "_")}_sale"
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
  count = terraform.workspace == "prod" ? 0 : 1
  # Buckets can only be a max of 63 characters long:
  # https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
  bucket        = "ccao-ci-${substr(terraform.workspace, 0, 33)}-glue-assets-us-east-1"
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

resource "aws_s3_bucket_versioning" "glue_assets" {
  count  = terraform.workspace == "prod" ? 0 : 1
  bucket = local.s3_bucket_glue_assets

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket" "data_warehouse" {
  count         = terraform.workspace == "prod" ? 0 : 1
  bucket        = "ccao-ci-${substr(terraform.workspace, 0, 30)}-data-warehouse-us-east-1"
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

resource "aws_s3_bucket_versioning" "data_warehouse" {
  count  = terraform.workspace == "prod" ? 0 : 1
  bucket = local.s3_bucket_data_warehouse

  versioning_configuration {
    status = "Enabled"
  }
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
  etag   = filemd5("${path.module}/glue/flagging_script_glue/flagging.py")
}

resource "aws_glue_job" "sales_val_flagging" {
  name              = local.glue_job_name
  role_arn          = var.iam_role_arn
  max_retries       = 0
  glue_version      = "3.0"
  execution_class   = "STANDARD"
  worker_type       = "G.2X"
  number_of_workers = 2

  command {
    name            = "glueetl"
    script_location = "s3://${aws_s3_object.sales_val_flagging.bucket}/${aws_s3_object.sales_val_flagging.key}"
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
    "--additional-python-modules" = "boto3==1.28.12,pandas==1.3.5,awswrangler==2.20.1,pyathena==2.25.2"
    "--commit_sha"                = var.commit_sha
    "--min_groups_threshold"      = "30"
    "--ptax_sd"                   = "1,1"
    "--sale_flag_table"           = "${local.athena_database_name}.flag"
  }
}

resource "aws_athena_database" "sale" {
  count         = terraform.workspace == "prod" ? 0 : 1
  name          = local.athena_database_name
  comment       = "Test sale database for the ${terraform.workspace} branch"
  bucket        = "ccao-athena-results-us-east-1"
  force_destroy = true
}

resource "aws_glue_crawler" "ccao_data_warehouse_sale_crawler" {
  name          = local.glue_crawler_name
  database_name = local.athena_database_name
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

resource "aws_glue_catalog_table" "sale_flag" {
  count         = terraform.workspace == "prod" ? 0 : 1
  name          = "flag"
  database_name = aws_athena_database.sale[0].id
  table_type    = "EXTERNAL_TABLE"
  parameters    = local.glue_table_sale_flag_parameters

  storage_descriptor {
    compressed                = false
    input_format              = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    location                  = "s3://${local.s3_bucket_data_warehouse}/sale/flag/"
    number_of_buckets         = -1
    output_format             = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    stored_as_sub_directories = false
    parameters                = local.glue_table_sale_flag_parameters

    columns {
      name = "meta_sale_document_num"
      type = "string"
    }

    columns {
      name = "rolling_window"
      type = "date"
    }

    columns {
      name = "sv_is_outlier"
      type = "bigint"
    }

    columns {
      name = "sv_is_ptax_outlier"
      type = "bigint"
    }

    columns {
      name = "ptax_flag_original"
      type = "boolean"
    }

    columns {
      name = "sv_is_heuristic_outlier"
      type = "bigint"
    }

    columns {
      name = "sv_outlier_type"
      type = "string"
    }

    columns {
      name = "run_id"
      type = "string"
    }

    columns {
      name = "version"
      type = "bigint"
    }

    ser_de_info {
      parameters = {
        "serialization.format" = "1"
      }
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
  }
}
