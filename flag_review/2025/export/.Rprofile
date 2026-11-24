# Scripts in this folder run with this folder as the working directory, but
# the renv environment is shared by every year and lives two levels up, in
# flag_review/. Point renv at it so the packages in flag_review/renv.lock load
Sys.setenv(RENV_PROJECT = normalizePath("../.."))
source("../../renv/activate.R")
