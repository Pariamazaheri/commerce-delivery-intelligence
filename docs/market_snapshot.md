# Samand public market snapshot

The independent acquisition example contains **50 unique public vehicle listings**
collected on **7 October 2026** from Bama. Every retained vehicle is a Samand with a
Solar Hijri production year greater than 1385. Listings from 1385 or earlier are
recorded as rejected observations in `collection_status.json`.

Required fields are price (Toman), mileage (km), body color, Solar Hijri production
year, manual/automatic transmission and seller description. Source URLs, collection
timestamps (UTC) and SHA-256 page fingerprints provide provenance. No seller name,
phone field or account credentials are collected. Phone-like strings in descriptions
are redacted. The Excel snapshot preserves Persian descriptions and numeric values.

Five prices are missing. Negotiable and installment-only amounts are not treated
as cash prices. Unusually low stated cash prices remain the seller's stated value;
the scraper does not silently replace them with estimates. Price units come from
the public page's Toman presentation; no Rial conversion is inferred.

## Refresh

From the installed repository:

```bash
python -m commerce_intelligence.scraping --target 50 --xlsx
```

This overwrites the snapshot with the new collection. It checks `robots.txt`,
waits 1.5 seconds between sequential requests, and only accesses public listing
and detail pages. It reads the observed Nuxt SSR structure; schema changes fail
closed. No login, private API, CAPTCHA bypass or hidden account access is used.
An incomplete run saves actual collected rows and exits with a nonzero status.

The included Excel was created from the same JSON observations and verified against
their identifiers and numeric fields. The portable runtime exporter uses openpyxl;
its text handling neutralizes formula-like seller descriptions.

## Sampling limits

The scraper visits the newest visible listings across model/trim filters until it
reaches 50 unique qualifying cars. It does not claim random or exhaustive sampling.
Filter ordering, recency, geography, duplicate sellers and advertising can bias the
snapshot. Prices are asking prices, not completed transactions. This demonstration
is excluded from the Olist model and supports no market-wide valuation claim.

Source: [Bama Samand listings](https://bama.ir/car/samand).
Third-party listing content is attributed to Bama and its listing authors; this
repository does not assert ownership or blanket redistribution rights over it.
