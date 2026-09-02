"""MCP Server — exposes chem.* tools via MCP (fallback to FastAPI tool endpoints if MCP lib unavailable)."""
from __future__ import annotations

import asyncio
import json
from typing import Any, List

from pharmforge.chem import fingerprint_similarity, predict_properties, validate_smiles
from pharmforge.rag import query_rag
from pharmforge.data.loader import load_molecules

# Try real MCP; fallback to simple JSON-RPC over stdio
try:
    from mcp.server import Server  # type: ignore
    from mcp.types import Tool, TextContent  # type: ignore
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

TOOLS = [
    {"name": "chem.search", "description": "Hybrid RAG search over chemical dataset. Args: {query: str, k: int}", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 5}}, "required": ["query"]}},
    {"name": "chem.similarity", "description": "Tanimoto similarity between two SMILES. Args: {smiles1, smiles2}", "inputSchema": {"type": "object", "properties": {"smiles1": {"type": "string"}, "smiles2": {"type": "string"}}, "required": ["smiles1", "smiles2"]}},
    {"name": "chem.properties", "description": "Predict ADME properties for SMILES. Args: {smiles}", "inputSchema": {"type": "object", "properties": {"smiles": {"type": "string"}}, "required": ["smiles"]}},
    {"name": "chem.validate", "description": "Validate SMILES string. Args: {smiles}", "inputSchema": {"type": "object", "properties": {"smiles": {"type": "string"}}, "required": ["smiles"]}},
    {"name": "chem.get_molecule", "description": "Get molecule by ID. Args: {id}", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
]

def handle_tool(name: str, arguments: dict) -> Any:
    if name == "chem.search":
        docs = query_rag(arguments.get("query", ""), k=arguments.get("k", 5))
        return [d.model_dump() for d in docs]
    elif name == "chem.similarity":
        s = fingerprint_similarity(arguments["smiles1"], arguments["smiles2"])
        return {"tanimoto": s}
    elif name == "chem.properties":
        return predict_properties(arguments["smiles"]).model_dump()
    elif name == "chem.validate":
        ok, err = validate_smiles(arguments["smiles"])
        return {"valid": ok, "error": err}
    elif name == "chem.get_molecule":
        mols = load_molecules()
        for m in mols:
            if m.id == arguments["id"]:
                return m.model_dump()
        return {"error": "not found"}
    else:
        return {"error": f"unknown tool {name}"}

# Real MCP server (if library present)
if MCP_AVAILABLE:
    server = Server("pharmforge")

    @server.list_tools()  # type: ignore
    async def _list_tools():
        return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]) for t in TOOLS]

    @server.call_tool()  # type: ignore
    async def _call_tool(name: str, arguments: dict):
        result = handle_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    def run_stdio():
        import mcp.server.stdio  # type: ignore
        async def _run():
            async with mcp.server.stdio.stdio_server() as (read, write):  # type: ignore
                await server.run(read, write, server.create_initialization_options())  # type: ignore
        asyncio.run(_run())
else:
    # Fallback JSON-RPC loop reading from stdin
    def run_stdio():
        import sys
        print(json.dumps({"tools": TOOLS}), flush=True)
        for line in sys.stdin:
            line=line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if req.get("method") == "list_tools":
                    print(json.dumps({"tools": TOOLS}), flush=True)
                elif req.get("method") == "call_tool":
                    name = req.get("params", {}).get("name")
                    args = req.get("params", {}).get("arguments", {})
                    res = handle_tool(name, args)
                    print(json.dumps({"result": res}), flush=True)
                else:
                    # direct tool call {tool, arguments}
                    if "tool" in req:
                        res = handle_tool(req["tool"], req.get("arguments", {}))
                        print(json.dumps(res), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    run_stdio()
