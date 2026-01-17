    def fetch_positions(self) -> List[PositionUpdate]:
        """終極同步版：確保消失的倉位也會被回報為 0"""
        res = []
        try:
            # 獲取所有持倉
            ps = self.exchange.fetch_positions()
            
            # 如果完全沒有持倉，回傳一個特殊標記或空列表
            if not ps:
                return []

            for p in ps:
                symbol = p.get("symbol", "")
                # 取得數量，如果抓不到就設為 0.0
                qty = abs(float(p.get("contracts", 0) or p.get("size", 0) or 0))
                sd = p.get("side", "").upper()
                
                if sd in ["LONG", "SHORT"]:
                    res.append(PositionUpdate(
                        symbol=symbol,
                        position_side=sd,
                        quantity=qty,
                        entry_price=float(p.get("entryPrice", 0)),
                        unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                        leverage=int(p.get("leverage", 1))
                    ))
            
            # 這裡不篩選 qty > 0，讓 qty 為 0 的數據也能傳出去
            return res
        except Exception as e:
            logger.error(f"[Bitget] 獲取持倉發生錯誤: {e}")
            return []
