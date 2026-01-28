# -*- coding: utf-8 -*-
"""
===================================
查看分析结果
===================================

用途：查询和显示数据库中保存的分析结果
"""

import argparse
import json
from datetime import datetime
from src.storage import get_db


def print_result(result: dict, verbose: bool = False):
    """打印单个分析结果"""
    print(f"\n{'='*80}")
    print(f"📊 {result['name']} ({result['code']})")
    print(f"{'='*80}")
    print(f"分析时间: {result['created_at']}")
    print(f"综合评分: {result['sentiment_score']}/100")
    print(f"操作建议: {result['operation_advice']}")
    print(f"趋势预测: {result['trend_prediction']}")
    print(f"置信度: {result['confidence_level']}")
    
    if verbose:
        print(f"\n--- 决策仪表盘 ---")
        if result['dashboard']:
            print(json.dumps(result['dashboard'], ensure_ascii=False, indent=2))
        
        print(f"\n--- 走势分析 ---")
        print(f"趋势分析: {result.get('trend_analysis', 'N/A')}")
        print(f"短期展望: {result.get('short_term_outlook', 'N/A')}")
        print(f"中期展望: {result.get('medium_term_outlook', 'N/A')}")
        
        print(f"\n--- 技术面 ---")
        print(f"技术分析: {result.get('technical_analysis', 'N/A')}")
        print(f"均线分析: {result.get('ma_analysis', 'N/A')}")
        print(f"成交量分析: {result.get('volume_analysis', 'N/A')}")
        print(f"形态分析: {result.get('pattern_analysis', 'N/A')}")
        
        print(f"\n--- 基本面 ---")
        print(f"基本面分析: {result.get('fundamental_analysis', 'N/A')}")
        print(f"行业地位: {result.get('sector_position', 'N/A')}")
        print(f"公司亮点: {result.get('company_highlights', 'N/A')}")
        
        print(f"\n--- 消息面 ---")
        print(f"新闻摘要: {result.get('news_summary', 'N/A')}")
        print(f"市场情绪: {result.get('market_sentiment', 'N/A')}")
        print(f"热点话题: {result.get('hot_topics', 'N/A')}")
        
        print(f"\n--- 综合分析 ---")
        print(f"分析摘要: {result.get('analysis_summary', 'N/A')}")
        print(f"关键要点: {result.get('key_points', 'N/A')}")
        print(f"风险提示: {result.get('risk_warning', 'N/A')}")
        print(f"买入理由: {result.get('buy_reason', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='查看分析结果')
    parser.add_argument('--code', type=str, help='股票代码（如 600519）')
    parser.add_argument('--limit', type=int, default=10, help='返回记录数（默认 10）')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    db = get_db()
    
    if args.code:
        print(f"查询股票 {args.code} 的最新分析结果...")
        result = db.get_latest_analysis_result(args.code)
        if result:
            print_result(result, verbose=args.verbose)
        else:
            print(f"未找到股票 {args.code} 的分析结果")
    else:
        print(f"查询最近 {args.limit} 条分析结果...")
        results = db.get_analysis_results(limit=args.limit)
        if results:
            for result in results:
                print_result(result, verbose=args.verbose)
        else:
            print("未找到任何分析结果")


if __name__ == "__main__":
    main()
