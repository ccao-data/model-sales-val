-- Every 2025 sale in a township that has a reviewed sale, one row per sale,
-- with how far its price sits from the typical price of similar sales.
--
-- sv_price_deviation is the number of standard deviations between the sale's
-- logged price and the mean logged price of its sales validation group (same
-- area, property type, and trailing 12 months). It comes from the most recent
-- version of the sale's row in sale.flag.
--
-- The three sale filters are the ones the residential model applies to its
-- sales
WITH reviewed_townships AS (
    SELECT DISTINCT sale.township_code
    FROM sale.flag_review AS review
    INNER JOIN default.vw_pin_sale AS sale
        ON review.doc_no = sale.doc_no
),

latest_version AS (
    SELECT
        meta_sale_document_num,
        MAX(version) AS version
    FROM sale.flag
    GROUP BY meta_sale_document_num
)

SELECT
    sale.doc_no,
    sale.pin,
    flag.sv_price_deviation AS price_deviation
FROM default.vw_pin_sale AS sale
INNER JOIN latest_version
    ON sale.doc_no = latest_version.meta_sale_document_num
INNER JOIN sale.flag AS flag
    ON latest_version.meta_sale_document_num = flag.meta_sale_document_num
    AND latest_version.version = flag.version
WHERE sale.year = '2025'
    AND sale.township_code IN (
        SELECT reviewed_townships.township_code FROM reviewed_townships
    )
    AND NOT sale.sale_filter_same_sale_within_365
    AND NOT sale.sale_filter_less_than_10k
    AND NOT sale.sale_filter_deed_type
    AND flag.sv_price_deviation IS NOT NULL
