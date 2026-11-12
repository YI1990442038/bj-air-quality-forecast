"""M6 应用层：FastAPI 服务 + 静态前端托管。

启动：
    .venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

端点（前缀 /api）：
    GET /api/stations            35 个建模站点（含经纬度）
    GET /api/overview            数据概览（覆盖/缺失率/时间轴完整度）
    GET /api/trend?station=&start=&end=   单站多污染物时间序列
    GET /api/replay?station=&h=&mode=      预测回放（双模式拦截）
    GET /api/comparison          模型对比（v4 指标/峰值/特征重要度/DL 结果）
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import data as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="北京空气质量 PM2.5 短期预测 · 演示系统", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/stations")
def api_stations():
    return {"stations": D.get_stations()}


@app.get("/api/overview")
def api_overview():
    return D.get_overview()


@app.get("/api/trend")
def api_trend(
    station: str = Query(..., description="站点名，如 东城东四"),
    start: str | None = Query(None, description="起始时间 YYYY-MM-DD"),
    end: str | None = Query(None, description="结束时间 YYYY-MM-DD"),
    limit: int = 2000,
):
    try:
        return D.get_trend(station, start, end, limit)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/replay")
def api_replay(
    station: str = Query(..., description="站点名"),
    h: int = Query(6, ge=1, le=6, description="预测跨度"),
    mode: str = Query("review", pattern="^(review|historical_replay)$"),
    replay_time: str | None = Query(None, description="replay 模式下的最晚目标时刻"),
    limit: int = 1500,
):
    try:
        return D.get_replay(station, h, mode, replay_time, limit)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/comparison")
def api_comparison():
    return D.get_comparison()


@app.get("/api/live")
def api_live():
    return D.get_live()


# 静态前端（Vue3 + ECharts），根路径
if os.path.isdir(STATIC):
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
