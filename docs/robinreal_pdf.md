## Making every real estate decision faster

## and smarter

**DATATHON 2026**


###### The Challenge

### User queries mix hard constraints with soft preferences

Your system needs to separate these two layers and combine them effectively.

```
QUERY 1
" 3 - room bright apartment in Zurich
under 2800 CHF with balcony , close
to public transport"
■ Hard ■ Soft
```
```
QUERY 2
"Bright family-friendly flat in
Winterthur , not too expensive, ideally
with parking"
Mostly soft preferences — harder to rank
```
```
QUERY 3
"Modern studio in Geneva for June
move-in , quiet area, nice views if
possible"
Conflicting: cheap vs. central, quiet vs. views
```
**■ Hard constraints** = must be respected **■ Soft preferences** = should influence ranking


###### Your Task

### Map a natural-language query to a ranked list of

### listings

```
STEP 1
Extract Hard Filters
Parse the query into strict
criteria: rooms, price, city,
features
```
##### →

```
STEP 2
Retrieve
Candidates
Query listings that satisfy
all hard constraints via the
```
##### API →

```
STEP 3
Rank by Relevance
Score soft preferences and
sort by overall relevance
```
##### →

```
OUTPUT
Ranked Results
Sorted listing IDs with
scores + explanations
```

###### The Data

### Listing Data Structure

```
IDENTITY & PROPERTY
listing_id 36272162
title "3.5 Zimmer Wohnung"
description full text
category "Wohnung"
marketing_type RENT
status ACTIVE / INACTIVE
```
```
LOCATION & GEO
street "38 Röhrliberg"
city "Cham"
postal_code "6330"
canton "ZG"
latitude / longitude 47.186 / 8.
```
```
PRICING & SIZE
price 2450 (CHF/month)
num_rooms 3.
living_area 97 m²
floor 2
available_date 2025 - 12 - 12
construction_year 1979
```
```
FEATURES (boolean flags)
HasBalconies ✓ HasBaths ✓ HasParkingIndoor ✓ HasWashingmachine ✓
HasElevator HasGarden HasPool HasFireplace PetsAllowed ...
Structured flags where available — not all listings have complete feature data
```
```
MEDIA
image_urls array of CDN links
local image bundle
Images useful for multimodal
ranking
```

###### Getting Started

### Your Starter Kit

```
DOCKER & COMPOSE
Full local dev environment. One command to spin
up all services.
docker compose up
```
```
HARD FILTER API
Pre-built filtering service. Extract constraints from
queries and filter listings.
rooms, price, city, features
```
```
AI SDK APP
Scaffolded app with Claude (provided by
Datathon) for LLM-powered query parsing.
Claude 4 Sonnet ready to go
```
```
MCP SERVER
Model Context Protocol server for tool-augmented
AI agent workflows.
connect your AI to live data
```
```
LISTING DATASET
Swiss rental listings hosted on AWS (provided by
Datathon). Structured data + images.
~thousands of listings on S
```
```
IMAGE BUNDLE
Property photos for multimodal analysis. Room
detection, quality scoring, visual ranking.
jpg, multiple per listing
```
```
💡 We recommend using AWS and Claude provided by Datathon — credits are included for all participants.
```
https://shorturl.at/NRPAw


```
DATA ENRICHMENT
```
#### Go Beyond the Dataset

Listings have basic data — enrich them with external sources to unlock stronger ranking signals.

**Transport & Commute**

- SBB API for travel times
- Public transport stops nearby
- Commute duration to key hubs
- Proximity to train stations

**Amenities & Services**

- Supermarkets, pharmacies, schools
- Restaurants & nightlife density
- Healthcare facilities nearby
- OpenStreetMap POI data (Nominatim)

**Geospatial Features**

- Geocoding via Nominatim
- Neighborhood classification
- Walkability & bike scores
- Elevation & noise levels

**Neighborhood Context**

- Average rents in the area
- Safety & crime statistics
- Demographic profiles
- Green space & parks ratio

**Image Analysis**

- Room brightness & natural light
- Modernity of interiors
- View quality from windows
- Space perception & layout

**Tex t E nric h m e n t**

- Extract features from descriptions
- Sentiment of listing language
- Keyword tagging (cozy, modern)
- Multilingual normalization

**EXAMPLE — WHY ENRICHMENT MATTERS**

```
"I want a quiet apartment near ETH with a short commute to the city center , modern-looking inside, with good schools nearby "
→ near ETH needs Geospatial short commute needs Transport modern-looking needs Image Analysis good schools needs Amenities
```

###### Example 1: Clear Hard Constraints

**User Query**

###### " 3.5-room bright apartment in Zurich under CHF 2800 with balcony "

**Bold** = hard filter Orange = soft preference

**Candidate Listings After Hard Filters** (city = Zürich, rooms = 3.5, price ≤ 2800, balcony = true)

```
Helle 3.5-Zi an der Wengistrasse
Wengistrasse 2, 8004 Zürich
```
###### CHF 2,650/mo

```
3.5 rooms · 85 m² · 2nd floor
Balcony ✓ Minergie ✓ Washer ✓
Bright Minergie-certified apartment in Kreis 4. Large
south-facing windows, modern kitchen, parquet floors.
⭐ bright + Minergie + central
```
```
Moderne Wohnung Oerlikon
Käferholzstrasse 42, 8057 Zürich
```
###### CHF 2,450/mo

```
3.5 rooms · 75 m² · 3rd floor
Balcony ✓ Washer ✓ Parkett ✓
Maisonette near Bucheggplatz. Rooftop terrace, close to
ETH Hönggerberg campus. Well connected.
🏘 maisonette + terrace
```
```
Gemütliche Wohnung Seebach
Hertensteinstrasse 8, 8052 Zürich
```
###### CHF 2,200/mo

```
3.5 rooms · 78 m² · 4th floor
Balcony ✓ Washer ✓ Parking ✓
Private terrace with access from all rooms. In-unit
washer/dryer. Near Seebach station. Quiet area.
🌳 quiet + parking + value
```

###### Example 1: Ranking Output

### How should these rank for "bright"?

#### #

**Helle 3.5-Zi, Wengistrasse**

**CHF 2,650** score: 0.
**Why this rank?**

- "Helle" (bright) in title — strong signal
- Minergie = large windows, energy-efficient glass
- South-facing, 2nd floor — maximum light
- Modern 2019 build with open floor plan

#### #

```
Moderne Wohnung, Oerlikon
CHF 2,450 score: 0.
Why this rank?
```
- Maisonette = windows on two levels
- Roof terrace — skylights likely
- Near ETH campus — well-maintained area
- No explicit “bright” signal in listing

#### #

```
Gemütliche Wohnung, Seebach
CHF 2,200 score: 0.
Why this rank?
```
- “Gemütlich” (cozy) ≠ bright
- 4th floor = good light potential
- Best price, but weaker brightness signal
- Parking + terrace still strong amenities

**Key insight:** "Bright" is a soft preference — your ranking system must interpret description text + metadata to score it


###### Example 2: Vague & Soft-Heavy

**User Query**

###### "Bright family-friendly flat in Winterthur , not too expensive, ideally with parking"

**Challenge:** Only "Winterthur" is a hard filter. Everything else is subjective. How does the system interpret "not too expensive"?

**System Must Interpret**

- **"not too expensive"** → below Winterthur median? bottom 30%? Infer from
    context: 3.5 rooms → ~CHF 1,600–2,450 range
- **"family-friendly"** → 3+ rooms, near schools/parks, quiet area, ground floor
    or elevator, pets allowed?
- **"bright"** → large windows, high floor, south-facing, or
    "hell"/"lichtdurchflutet" in description
- **"ideally with parking"** → bonus, not a hard requirement. HasParkingIndoor
    flag, or mention in description


###### Example 3: Conflicting Preferences

**User Query**

###### "Modern apartment in Basel , cheap but central, quiet with nice views"

**Conflict:** "Cheap" and "central" are inversely correlated. "Quiet" and "central" often conflict. How does your system balance tradeoffs?

```
Strategy A
Prioritize Price
2.5-Zi Kleinbasel, Klybeckstr.
CHF 1,350/mo
48 m² · 2nd floor · 1995 build
✓ Cheapest option
✓ Central Kleinbasel location
✗ Street noise (Klybeck is busy)
✗ Older, not “modern”
✗ No views to speak of
```
```
Strategy B
Balance All Signals
3.5-Zi Gundeldingen, Güterstr.
CHF 1,950/mo
72 m² · 5th floor · 2021 build
✓ Modern new build
✓ 5th floor = views + quiet
✓ 10 min to SBB by tram
✗ Not the cheapest
✗ “Central” is debatable
```
```
Strategy C
Prioritize Lifestyle
3.5-Zi St. Alban, Münsterplatz
CHF 2,800/mo
85 m² · 3rd floor · Renovated
✓ Premium central location
✓ Rhine views, historic quiet
✓ High-end renovation
✗ Not cheap at all
✗ May exceed user budget
```

**EXAMPLE — IMAGE ANALYSIS FOR RANKING**

#### What Can You Learn from Photos?

Vision models can extract soft ranking signals that text alone misses

```
☀ Brightness & Light
Detect natural light levels, window size, south-facing
rooms, time-of-day photography bias
```
```
✨ Interior Modernity
Classify kitchen/bathroom age, identify recent
renovations, detect style (modern vs. traditional)
```
```
📐 Space & Layout
Estimate perceived spaciousness, detect open floor
plans, measure visual clutter
```
```
🏔 Views & Surroundings
Identify window views (greenery, cityscape,
mountain), balcony presence, outdoor spaces
```
```
How to use image signals
Combine vision scores with text-based relevance for multimodal ranking. A listing described as "bright" that also
shows dark photos should be penalized.
```

###### Approaches

### Solution Directions

```
LLM QUERY DECOMPOSITION
Use GPT-4o to parse natural language into
structured filters + soft preferences. Handle
ambiguity, infer missing constraints.
```
💡

```
Prompt engineering is key
```
```
HYBRID SEARCH
Combine keyword search (BM25) with semantic
embeddings. Structured filters narrow candidates,
embeddings rank by meaning.
```
```
💡
```
```
Best of both worlds
```
```
EMBEDDING-BASED RANKING
Embed queries and listing descriptions into shared
vector space. Rank by cosine similarity. Fine-tune
for real estate domain.
```
```
💡
```
```
Handles soft preferences well
```
```
MULTIMODAL SIGNALS
Use listing images to assess brightness,
modernity, views. Vision models can extract
features invisible in structured data.
```
💡

```
Images reveal what text hides
```
```
GEOSPATIAL ENRICHMENT
Use lat/lng to compute distances to transit,
schools, parks. Enrich listings with neighborhood
characteristics.
```
```
💡
```
```
Great for “central” or “quiet”
```
```
LEARN-TO-RANK
Train a model on pairwise preferences. Combine
all signals (text, features, geo, images) into a
unified relevance score.
```
```
💡
```
```
Advanced but powerful
```

**BONUS**

### User Preference History

Personalize ranking using past interaction signals. This is optional but will impress the jury.

**Interaction Signals**

- **Favorites / saves** — explicit positive signal
- **Click-through rate** — which listings get opened
- **Dwell time** — time spent on a listing page
- **Query refinement patterns** — how users narrow down
- **Dismissals / skips** — implicit negative signal
Build a user profile that shifts ranking weights over time. A user who always saves modern
apartments should see them ranked higher.

```
Example Scenario
A user has previously:
```
- Saved 5 apartments with Minergie label
- Spent 3x more time on listings with balcony photos
- Skipped all listings above CHF 2,
- Always searched in Kreis 4– 6
**→ Inferred profile:**
Eco-conscious, outdoor living, budget-sensitive, urban core preference. Boost
Minergie + balcony + central + affordable in ranking weights.


###### Judging

### Evaluation & Judging

**Peer Review – Hidden and public test set**

- **Peer feedback** — Other teams evaluate your output
- **Prepared Public queries**
- **Hidden Test queries**
- **Hard-Filter Precision** — Do results actually satisfy hard
    constraints?

**Jury & Peer Review**

- **Technical depth** — Quality of architecture and approach
- **Feature width** — Variety of different approaches and
    considerations
- **Creativity** — Novel ideas, unexpected data usage
- **Demo quality** — Working prototype, clear presentation
- **Failure analysis** — Show where your system breaks
- **Peer feedback** — Other teams evaluate your output

**Deliverables**
① Working prototype ② Technical write-up ③ Live demo ④ Ranking logic explanation
**Submission:** Live GitHub repo with running code & presentation


