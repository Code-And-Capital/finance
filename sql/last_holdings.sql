SELECT *
FROM [dbo].[holdings]
WHERE [DATE] = (
    SELECT MAX([DATE])
    FROM [dbo].[holdings]
)
ORDER BY WEIGHT DESC