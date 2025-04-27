import pandas as pd
import io
from fastapi import HTTPException, UploadFile
from app.models.models import CalculationRequest

async def process_excel_file(file: UploadFile) -> CalculationRequest:
    """
    Process the uploaded Excel file and extract the data needed for calculation
    """
    try:
        # Read the Excel file
        contents = await file.read()
        excel_data = pd.read_excel(io.BytesIO(contents), sheet_name=None)

        # Get the first sheet
        first_sheet_name = list(excel_data.keys())[0]
        first_sheet = excel_data[first_sheet_name]

        # Check if this is the single-sheet format (has Day columns)
        day_columns = [col for col in first_sheet.columns if str(col).startswith("Day ")]

        if day_columns:
            # This is the single-sheet format
            df = first_sheet

            # Validate required columns
            if "Price" not in df.columns or "Support Cost" not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Excel file must contain 'Price' and 'Support Cost' columns"
                )

            # Extract price and support_cost data
            price_data = df["Price"].tolist()
            support_cost_data = df["Support Cost"].tolist()

            # Extract daily orders data
            daily_orders_data = []
            for day_col in sorted(day_columns, key=lambda x: int(x.split(" ")[1])):
                daily_orders_data.append(df[day_col].tolist())

            n_days = len(daily_orders_data)
        else:
            # This is the traditional format with separate Services and Orders sheets
            if "Services" not in excel_data:
                raise HTTPException(status_code=400, detail="Excel file must contain a 'Services' sheet")

            # Extract service data
            services_df = excel_data["Services"]

            # Validate required columns
            if not all(col in services_df.columns for col in ["price", "support_cost"]):
                raise HTTPException(
                    status_code=400,
                    detail="Services sheet must contain 'price' and 'support_cost' columns"
                )

            # Extract price and support_cost data
            price_data = services_df["price"].tolist()
            support_cost_data = services_df["support_cost"].tolist()

            # Check if Orders sheet exists
            if "Orders" not in excel_data:
                raise HTTPException(status_code=400, detail="Excel file must contain an 'Orders' sheet")

            # Extract orders data
            orders_df = excel_data["Orders"]
            if orders_df.shape[1] < len(price_data):
                raise HTTPException(
                    status_code=400,
                    detail=f"Orders sheet must contain at least {len(price_data)} columns for services"
                )

            # Convert orders data to the required format
            daily_orders_data = orders_df.iloc[:, :len(price_data)].values.tolist()
            n_days = len(daily_orders_data)

        # Create and return the calculation request
        return CalculationRequest(
            price=price_data,
            support_cost=support_cost_data,
            daily_orders=daily_orders_data,
            n_days=n_days
        )

    except Exception as e:
        error_detail = f"Error processing Excel file: {str(e)}"
        raise HTTPException(status_code=400, detail=error_detail)
