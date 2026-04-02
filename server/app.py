"""FastAPI application entrypoint for the ATC optimization environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required to run this environment. Install dependencies first."
    ) from exc

try:
    from ..models import ATCOptimizationAction, ATCOptimizationObservation
    from .atc_environment import ATCOptimizationEnvironment
except ImportError:
    from models import ATCOptimizationAction, ATCOptimizationObservation
    from server.atc_environment import ATCOptimizationEnvironment


app = create_app(
    ATCOptimizationEnvironment,
    ATCOptimizationAction,
    ATCOptimizationObservation,
    env_name="atc_env",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server directly."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
