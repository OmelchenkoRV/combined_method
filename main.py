from fastapi import FastAPI, File, UploadFile, HTTPException
from app.models.models import CombinedMethodResponse
from app.utils import process_excel_file
from app.services.mopso import MOPSO
from app.services.report import generate_detailed_report

app = FastAPI(
    title="Service Package Optimizer",
    version="1.0.0"
)

@app.post(
    "/api/combined-method",
    response_model=CombinedMethodResponse
)
async def combined_method_endpoint(
    file: UploadFile = File(...),
    num_providers: int = 3
):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Only Excel files supported")

    req = await process_excel_file(file)
    # Initialize & run
    mopso = MOPSO(
        req,
        num_providers=num_providers,
        num_particles=50,
        max_iter=50
    )
    archive = mopso.optimize()
    best = mopso.get_best_compromise_solution()
    report = generate_detailed_report(mopso, best)

    pareto = [[float(p.current_objectives[0]), float(p.current_objectives[1])] for p in archive]
    return CombinedMethodResponse(
        service_allocation=report['service_allocation'],
        provider_analysis=report['provider_analysis'],
        financial_summary=report['financial_summary'],
        pareto_front=pareto
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)