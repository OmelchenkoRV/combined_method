# Combined Method API

This API implements a combined method for solving subproblems for providers and IT companies based on the probabilistic method, followed by improvement based on a heuristic procedure for the total solution of subproblems.

## Overview

The combined method consists of seven stages as described in README1.md:
1. Solving the subproblem for an IT company based on the probabilistic method
2. Solving the subproblem for providers based on the probabilistic method
3. Searching for provider solutions based on the IT company solution from stage 1
4. Searching for IT company solution based on provider solutions from stage 2
5. Improvement based on heuristic procedure for the total solution (IT company to providers)
6. Improvement based on heuristic procedure for the total solution (providers to IT company)
7. Selection of the best solution from stages 5 and 6

## API Endpoint

The API exposes a single endpoint:

- **POST /api/combined-method**: Accepts an Excel file with service and order data, processes it using the combined method, and returns the optimized solution.

### Request Parameters

- `file`: An Excel file containing the required data (required)
- `num_providers`: Number of IT service providers to consider (optional, default: 3)

### Excel File Format

The Excel file must contain two sheets:

1. **Services**: Contains service information with the following columns:
   - `service_id`: Unique identifier for each service
   - `price`: Price of the service
   - `support_cost`: Support cost for the service

2. **Orders**: Contains daily orders for each service
   - Each row represents a day
   - Each column represents a service (in the same order as the Services sheet)

### Response

The API returns a JSON response with the following structure:

```json
{
  "service_allocation": [
    {
      "service_id": 0,
      "price": 100.0,
      "support_cost": 40.0,
      "avg_daily_orders": 10.0,
      "allocated_to_providers": [
        {
          "provider_id": 0,
          "price": 95.0,
          "discount_rate": 0.1,
          "effective_price": 85.5,
          "support_cost": 38.0,
          "profit": 475.0
        }
      ]
    }
  ],
  "provider_analysis": [
    {
      "provider_id": 0,
      "services": [
        {
          "service_id": 0,
          "revenue": 855.0,
          "cost": 380.0,
          "profit": 475.0
        }
      ],
      "total_revenue": 855.0,
      "total_cost": 380.0,
      "total_profit": 475.0
    }
  ],
  "financial_summary": {
    "it_company_profit": 1000.0,
    "total_provider_profit": 1500.0,
    "system_total_profit": 2500.0
  },
  "pareto_front": [
    [900.0, 1400.0],
    [950.0, 1450.0],
    [1000.0, 1500.0]
  ]
}
```

## Deployment

### Local Deployment

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the FastAPI application:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. Test the API using the provided test script:
   ```
   python test_api.py
   ```

### Azure Deployment

The repository includes a GitHub Actions workflow in `azure/deploy.yml` that automates the deployment to Azure Functions. To use it:

1. Create an Azure Container Registry
2. Create an Azure Function App with Docker container support
3. Configure the following secrets in your GitHub repository:
   - `REGISTRY_LOGIN_SERVER`: The login server for your Azure Container Registry
   - `REGISTRY_USERNAME`: The username for your Azure Container Registry
   - `REGISTRY_PASSWORD`: The password for your Azure Container Registry
   - `AZURE_CREDENTIALS`: The credentials for Azure login (can be obtained using Azure CLI)

4. Push to the main branch to trigger the deployment

## Implementation Details

The implementation uses:
- FastAPI for the API framework
- Pandas and openpyxl for Excel file processing
- NumPy and Matplotlib for numerical computations and visualization
- Multi-Objective Particle Swarm Optimization (MOPSO) for the probabilistic method
- A custom heuristic procedure for solution improvement
