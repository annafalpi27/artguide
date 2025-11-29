run-api:
	@echo "🧹 Cleaning old images..."
	cd api && docker compose down --rmi all --volumes --remove-orphans || true
	@echo "🔨 Building and starting API..."
	cd api && docker compose up -d --build
	@echo "✅ API is running at http://localhost:7005"
	@echo "📚 API docs at http://localhost:7005/docs"
	@echo "📋 View logs with: make logs"

logs:
	cd api && docker compose logs -f

run_dashboard:
	@ export PYTHONPATH=.
	streamlit run dashboard/main.py --server.port 8501

