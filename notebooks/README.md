# SEM Campaign Optimizer - Complete Guide


### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Campaign
Edit the config.yaml file
```yaml

inputs:
  brand_website: "url of brand website"
  competitor_website: "url of competitor website" 
  service_locations: ["New York", "Los Angeles"]
  budgets:
    shopping_ads: 5000
    search_ads: 8000  
    pmax_ads: 2000
```

### 3. Run the Optimizer
```bash
python run_sem_optimizer.py
```


##  Project Structure

```
notebooks/
├── run_sem_optimizer.py      # Main execution script
├── sem_optimizer.py          # Core SEM optimization logic
├── keyword_planner.py        # Enhanced keyword research
├── campaign_builder.py       # Advanced campaign structuring
├── config.yaml              # Your configuration file
└── README.md               # This file
```

