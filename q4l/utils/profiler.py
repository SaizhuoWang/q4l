from typing import Dict

import psutil


class ProcessNode:
    """Represents a node in the process tree with memory details.

    Parameters
    ----------
    process : psutil.Process
        A process instance for which the node should be created.

    Attributes
    ----------
    pid : int
        Process ID of the process.
    memory_info : psutil._pslinux.pfullmem
        Memory information of the process.
    children : List[ProcessNode]
        List of child processes as `ProcessNode` objects.

    """

    def __init__(self, process: psutil.Process):
        self.pid = process.pid
        self.memory_info = process.memory_info()
        self.children = [ProcessNode(child) for child in process.children()]

    def get_memory_summary(self) -> Dict[str, int]:
        """Retrieve memory summary for the process and its children.

        Returns
        -------
        dict
            Memory details with keys 'shr', 'virt', and 'res'.

        """
        total_memory = {
            "shr": self.memory_info.shared,
            "virt": self.memory_info.vms,
            "res": self.memory_info.rss,
        }

        for child in self.children:
            child_summary = child.get_memory_summary()
            total_memory["shr"] += child_summary["shr"]
            total_memory["virt"] += child_summary["virt"]
            total_memory["res"] += child_summary["res"]

        return total_memory


def format_memory(value: int) -> str:
    """Convert bytes to a human-readable format.

    Parameters
    ----------
    value : int
        Memory value in bytes.

    Returns
    -------
    str
        Human-readable representation of the memory value.

    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def display_memory_tree(
    node: ProcessNode, level: int = 0, prefix: str = ""
) -> str:
    """Generate a string representing memory details of a process and its
    children in a tree structure.

    Parameters
    ----------
    node : ProcessNode
        Root node of the process tree to display.
    level : int, optional
        Level of the node in the tree, by default 0.
    prefix : str, optional
        Prefix string for tree representation, by default ''.

    Returns
    -------
    str
        String representation of the process memory details in a tree structure.

    """
    lines = []

    own_shr = format_memory(node.memory_info.shared)
    own_virt = format_memory(node.memory_info.vms)
    own_res = format_memory(node.memory_info.rss)

    lines.append(
        f"{prefix}PID: {node.pid} (Own) | Shared: {own_shr} | Virtual: {own_virt} | Resident: {own_res}"
    )

    if node.children:
        aggregated_memory = node.get_memory_summary()
        aggregated_shr = format_memory(aggregated_memory["shr"])
        aggregated_virt = format_memory(aggregated_memory["virt"])
        aggregated_res = format_memory(aggregated_memory["res"])

        lines.append(
            f"{prefix}PID: {node.pid} (Aggregated) | Shared: {aggregated_shr} | Virtual: {aggregated_virt} | Resident: {aggregated_res}"
        )

        # Adjust prefix for children
        for i, child in enumerate(node.children):
            if i < len(node.children) - 1:
                child_prefix = prefix + "│   "
                joiner = "├── "
            else:
                child_prefix = prefix + "    "
                joiner = "└── "

            lines.append(
                display_memory_tree(
                    child, level + 1, prefix=child_prefix + joiner
                )
            )

    return "\n".join(lines)
