WITH reviewed_townships AS (
    SELECT DISTINCT sale.township_code
    FROM sale.flag_review AS review
    INNER JOIN default.vw_pin_sale AS sale
        ON review.doc_no = sale.doc_no
),

changes AS (
    SELECT
        before.pin,
        MIN(before.township_code) AS township_code,
        -- Fields reported one by one
        BOOL_OR(before.char_rooms IS DISTINCT FROM after.char_rooms)
            AS rooms_changed,
        BOOL_OR(before.char_fbath IS DISTINCT FROM after.char_fbath)
            AS full_baths_changed,
        BOOL_OR(before.char_beds IS DISTINCT FROM after.char_beds)
            AS beds_changed,
        BOOL_OR(before.char_air IS DISTINCT FROM after.char_air)
            AS central_air_changed,
        BOOL_OR(before.char_bldg_sf IS DISTINCT FROM after.char_bldg_sf)
            AS bldg_sf_changed,
        BOOL_OR(before.char_bsmt_fin IS DISTINCT FROM after.char_bsmt_fin)
            AS basement_finish_changed,
        BOOL_OR(
            before.char_recent_renovation
            IS DISTINCT FROM after.char_recent_renovation
        ) AS recent_renovation_changed,
        BOOL_OR(before.char_hbath IS DISTINCT FROM after.char_hbath)
            AS half_baths_changed,
        BOOL_OR(before.class IS DISTINCT FROM after.class)
            AS class_changed,
        BOOL_OR(before.char_type_resd IS DISTINCT FROM after.char_type_resd)
            AS stories_changed,
        BOOL_OR(before.char_cnst_qlty IS DISTINCT FROM after.char_cnst_qlty)
            AS construction_quality_changed,
        BOOL_OR(before.char_attic_fnsh IS DISTINCT FROM after.char_attic_fnsh)
            AS attic_finish_changed,
        BOOL_OR(before.char_bsmt IS DISTINCT FROM after.char_bsmt)
            AS basement_type_changed,
        BOOL_OR(before.char_ext_wall IS DISTINCT FROM after.char_ext_wall)
            AS exterior_wall_changed,
        BOOL_OR(before.char_apts IS DISTINCT FROM after.char_apts)
            AS apartments_changed,
        -- Fields that only count toward "any characteristic changed"
        BOOL_OR(
            before.char_gar1_size IS DISTINCT FROM after.char_gar1_size
            OR before.char_gar1_att IS DISTINCT FROM after.char_gar1_att
            OR before.char_gar1_cnst IS DISTINCT FROM after.char_gar1_cnst
            OR before.char_attic_type IS DISTINCT FROM after.char_attic_type
            OR before.char_heat IS DISTINCT FROM after.char_heat
            OR before.char_porch IS DISTINCT FROM after.char_porch
            OR before.char_frpl IS DISTINCT FROM after.char_frpl
            OR before.char_roof_cnst IS DISTINCT FROM after.char_roof_cnst
        ) AS other_field_changed
    FROM default.vw_card_res_char AS before
    INNER JOIN default.vw_card_res_char AS after
        ON before.pin = after.pin
        AND before.card = after.card
        AND after.year = '2026'
    WHERE before.year = '2025'
        AND before.township_code IN (
            SELECT reviewed_townships.township_code FROM reviewed_townships
        )
    GROUP BY before.pin
),

trained AS (
    SELECT DISTINCT meta_pin AS pin
    FROM model.training_data
    WHERE run_id = '2026-07-30-ecstatic-carly'
        AND year = '2025'
),

sold AS (
    SELECT DISTINCT pin
    FROM default.vw_pin_sale
    WHERE year = '2025'
),

valued AS (
    SELECT DISTINCT meta_pin AS pin
    FROM model.assessment_card
    WHERE run_id = '2026-07-30-ecstatic-carly'
        AND year = '2026'
)

SELECT
    changes.*,
    trained.pin IS NOT NULL AS trained_on_2025_sale,
    sold.pin IS NOT NULL AS sold_in_2025,
    valued.pin IS NOT NULL AS valued_by_2026_model
FROM changes
LEFT JOIN trained
    ON changes.pin = trained.pin
LEFT JOIN sold
    ON changes.pin = sold.pin
LEFT JOIN valued
    ON changes.pin = valued.pin
