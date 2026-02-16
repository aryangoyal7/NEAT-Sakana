from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InnovationTracker:
    next_innovation: int
    next_node_id: int
    conn_innov: dict[tuple[int, int], int] = field(default_factory=dict)
    split_innov: dict[tuple[int, int], tuple[int, int, int]] = field(default_factory=dict)

    def get_connection_innovation(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self.conn_innov:
            self.conn_innov[key] = self.next_innovation
            self.next_innovation += 1
        return self.conn_innov[key]

    def get_split(self, src: int, dst: int) -> tuple[int, int, int]:
        key = (src, dst)
        if key not in self.split_innov:
            node_id = self.next_node_id
            self.next_node_id += 1
            in_innov = self.get_connection_innovation(src, node_id)
            out_innov = self.get_connection_innovation(node_id, dst)
            self.split_innov[key] = (node_id, in_innov, out_innov)
        return self.split_innov[key]
