-- Every sale that a valuations analyst reviewed, one row per sale, with the
-- flags the analyst set and the PIN that sold. All reviewed sales are from
-- 2025.
--
-- The flags come from analyst workbooks. A blank workbook cell is stored as
-- FALSE (or 'no' for has_characteristic_change), so FALSE does not always mean
-- the analyst answered no.
SELECT
    review.doc_no,
    sale.pin,
    review.is_arms_length,
    review.is_flip,
    review.has_class_change,
    review.has_characteristic_change,
    review.requires_field_check,
    review.work_drawer,
    review.source_file
FROM sale.flag_review AS review
INNER JOIN default.vw_pin_sale AS sale
    ON review.doc_no = sale.doc_no
