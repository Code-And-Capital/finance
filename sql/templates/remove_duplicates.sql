WITH dedup AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, date
            ORDER BY (SELECT NULL)
        ) AS rn
    FROM [dbo].[prices]
)
DELETE FROM dedup
WHERE rn > 1;