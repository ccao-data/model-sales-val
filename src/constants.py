import yaml

# It's helpful to factor these tables out into shared constants because we often
# need to switch to dev tables for testing
LOCATION_NEIGHBORHOOD_GROUP_TABLE = "location.neighborhood_group"
DEFAULT_CARD_RES_CHAR_TABLE = "default.vw_card_res_char"
DEFAULT_VW_PIN_CONDO_CHAR_TABLE = "default.vw_pin_condo_char"
DEFAULT_VW_PIN_SALE_TABLE = "default.vw_pin_sale"
DEFAULT_VW_PIN_UNIVERSE_TABLE = "default.vw_pin_universe"

with open("src/inputs.yaml", "r") as stream:
    INPUTS = yaml.safe_load(stream)
