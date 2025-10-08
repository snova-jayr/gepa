# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any, Literal

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from gepa.proposer.reflective_mutation.base import (
    BatchSampler,
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)

import json
import re
import os

def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match new cheatsheet sections
    slug_map = {
        "strategies_and_hard_rules": "shr",
        "hard_rules": "hr",
        "strategies_and_insights": "si",
        "apis_to_use_for_specific_information": "api",
        "useful_code_snippets_and_templates": "code",
        "code_snippets_and_templates": "code",
        "common_mistakes_and_correct_strategies": "cms",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics_and_workflows": "psw",
        "problem_solving_heuristics": "prob",
        "verification_checklist": "vc",
        "troubleshooting_and_pitfalls": "ts",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])

def parse_cheatsheet_line(line):
    """Parse a single cheatsheet line to extract components.

    Supports both formats:
    1) "[id] helpful=X harmful=Y :: content"
    2) "[id] content" (counts default to 0)
    """
    text = line.strip()
    # New/primary format with counts
    pattern_full = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    match = re.match(pattern_full, text)
    if match:
        return {
            'id': match.group(1),
            'helpful': int(match.group(2)),
            'harmful': int(match.group(3)),
            'content': match.group(4),
            'raw_line': line
        }
    # Fallback simple format without counts
    pattern_simple = r'\[([^\]]+)\]\s*(.*)'
    match2 = re.match(pattern_simple, text)
    if match2:
        return {
            'id': match2.group(1),
            'helpful': 0,
            'harmful': 0,
            'content': match2.group(2).strip(),
            'raw_line': line
        }
    return None

def get_next_global_id(cheatsheet_text):
    """Extract highest global ID and return next one"""
    max_id = 0
    lines = cheatsheet_text.strip().split('\n')
    
    for line in lines:
        parsed = parse_cheatsheet_line(line)
        if parsed:
            # Extract numeric part from ID
            id_match = re.search(r'-(\d+)$', parsed['id'])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)
    
    return max_id + 1


def format_cheatsheet_line(bullet_id, helpful, harmful, content):
    """Format a bullet into cheatsheet line format (counts removed)."""
    return f"[{bullet_id}] {content}"

def update_bullet_counts(cheatsheet_text, bullet_tags):
    """Update helpful/harmful counts based on tags (Counter layer)"""
    lines = cheatsheet_text.strip().split('\n')
    updated_lines = []
    
    # Create tag lookup - handle both old and new formats
    tag_map = {}
    if isinstance(bullet_tags, list) and len(bullet_tags) > 0:
        for tag in bullet_tags:
            if isinstance(tag, dict):
                # Handle both 'id' and 'bullet' keys for backwards compatibility
                bullet_id = tag.get('id') or tag.get('bullet', '')
                tag_value = tag.get('tag', 'neutral')
                if bullet_id:
                    tag_map[bullet_id] = tag_value
    
    if not tag_map:
        print("Warning: No valid bullet tags found to update counts")
        return cheatsheet_text
    
    for line in lines:
        if line.strip().startswith('#') or not line.strip():
            # Preserve section headers and empty lines
            updated_lines.append(line)
            continue
            
        parsed = parse_cheatsheet_line(line)
        # Counts have been removed from the cheatsheet; keep lines unchanged
        if parsed and parsed['id'] in tag_map:
            updated_lines.append(format_cheatsheet_line(parsed['id'], 0, 0, parsed['content']))
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)


def apply_curator_operations(cheatsheet_text, operations, next_id):
    """
    Apply curator operations to cheatsheet
    
    TODO: Future Operations (not implemented yet)
    - UPDATE: Rewrite existing bullets to be more accurate or comprehensive
    - MERGE: Combine related bullets into stronger ones  
    - CREATE_META: Add high-level strategy sections
    - DELETE: Remove outdated or incorrect bullets (if needed)
    """
    lines = cheatsheet_text.strip().split('\n')
    
    # Build section map
    sections = {}
    current_section = "general"
    section_line_map = {}  # Track which line each section header is on
    # import pdb
    # pdb.set_trace()
    for i, line in enumerate(lines):
        if line.strip().startswith('##'):
            # Extract section name and normalize it
            section_header = line.strip()[2:].strip()
            # Normalize: lowercase, spaces->_, &->and, strip trailing ':'
            normalized = section_header.lower().replace(' ', '_').replace('&', 'and').rstrip(':')
            current_section = normalized
            section_line_map[current_section] = i
            if current_section not in sections:
                sections[current_section] = []
        elif line.strip():
            sections[current_section].append((i, line))
    
    # Process operations
    bullets_to_add = []
    
    for op in operations:
        op_type = op['type']
        
        # TODO: Future operation types (not implemented yet)
        # elif op_type == 'UPDATE':
        #     bullet_id = op.get('bullet_id', '')
                    #     new_content = op.get('content', '')
            #     bullets_to_update[bullet_id] = new_content
        # elif op_type == 'MERGE':
        #     source_ids = op.get('source_ids', [])
        #     bullets_to_delete.update(source_ids)
        #     # Add merged bullet logic here
        # elif op_type == 'CREATE_META':
        #     section_name = op.get('section_name', 'META_STRATEGIES')
        #     # Add meta section creation logic here
        
        if op_type == 'ADD':
            # Normalize section name from operation
            section_raw = op.get('section', 'general')
            section = section_raw.lower().replace(' ', '_').replace('&', 'and').rstrip(':')
            
            # Check if section exists, if not use 'others'
            if section not in sections and section != 'general':
                print(f"Warning: Section '{section_raw}' not found, adding to OTHERS")
                section = 'others'
            
            slug = get_section_slug(section)
            new_id = f"{slug}-{next_id:05d}"
            next_id += 1
            
            content = op.get('content', '')
            
            new_line = format_cheatsheet_line(new_id, 0, 0, content)
            bullets_to_add.append((section, new_line))
            print(f"  Added bullet {new_id} to section {section}")
            

    
    # Rebuild cheatsheet
    new_lines = []
    for line in lines:
        parsed = parse_cheatsheet_line(line)
        if parsed:
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Add new bullets to appropriate sections
    final_lines = []
    current_section = None
    
    for line in new_lines:
        if line.strip().startswith('##'):
            # Before moving to new section, add any bullets for current section
            if current_section:
                section_adds = [b for s, b in bullets_to_add if s == current_section]
                final_lines.extend(section_adds)
                # Clear added bullets
                bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
            
            section_header = line.strip()[2:].strip()
            current_section = section_header.lower().replace(' ', '_').replace('&', 'and').rstrip(':')
        final_lines.append(line)
    
    # Add remaining bullets to current section
    if current_section:
        section_adds = [b for s, b in bullets_to_add if s == current_section]
        final_lines.extend(section_adds)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
    
    # If there are still bullets to add (for sections that don't exist), add them to OTHERS
    if bullets_to_add:
        print(f"Warning: {len(bullets_to_add)} bullets have no matching section, adding to OTHERS")
        others_bullets = [b for s, b in bullets_to_add]
        # Find OTHERS section
        others_idx = -1
        for i, line in enumerate(final_lines):
            if line.strip() == "## OTHERS":
                others_idx = i
                break
        
        if others_idx >= 0:
            # Insert after OTHERS header
            for i, bullet in enumerate(others_bullets):
                final_lines.insert(others_idx + 1 + i, bullet)
        else:
            # Append to end
            final_lines.extend(others_bullets)
    
    return '\n'.join(final_lines), next_id

def get_cheatsheet_stats(cheatsheet_text):
    """Generate statistics about the cheatsheet"""
    lines = cheatsheet_text.strip().split('\n')
    stats = {
        'total_bullets': 0,
        'by_section': {}
    }
    
    current_section = 'general'
    
    for line in lines:
        if line.strip().startswith('##'):
            current_section = line.strip()[2:].strip()
            continue
            
        parsed = parse_cheatsheet_line(line)
        if parsed:
            stats['total_bullets'] += 1
            
            if current_section not in stats['by_section']:
                stats['by_section'][current_section] = {'count': 0}
            
            stats['by_section'][current_section]['count'] += 1
    
    return stats

def extract_json_from_text(text, json_key=None):
    """Extract JSON object from text, handling various formats"""
    try:
        # First, try to parse the entire response as JSON (JSON mode)
        try:
            result = json.loads(text.strip())
            return result
        except json.JSONDecodeError:
            pass
        
        # Fallback: Look for ```json blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Try each match until we find valid JSON
            for match in matches:
                try:
                    json_str = match.strip()
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Improved JSON extraction using balanced brace counting
        # This handles deeply nested structures better
        def find_json_objects(text):
            """Find JSON objects using balanced brace counting"""
            json_objects = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    # Found start of potential JSON object
                    brace_count = 1
                    start = i
                    i += 1
                    
                    while i < len(text) and brace_count > 0:
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                        elif text[i] == '"':
                            # Handle quoted strings to avoid counting braces inside strings
                            i += 1
                            while i < len(text) and text[i] != '"':
                                if text[i] == '\\':
                                    i += 1  # Skip escaped character
                                i += 1
                        i += 1
                    
                    if brace_count == 0:
                        # Found complete JSON object
                        json_candidate = text[start:i]
                        json_objects.append(json_candidate)
                else:
                    i += 1
            
            return json_objects
        
        # Find all potential JSON objects
        json_objects = find_json_objects(text)
        
        for json_str in json_objects:
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        print(f"Failed to extract JSON: {e}")
        if len(text) > 500:
            print(f"Raw content preview:\n{text[:500]}...")
        else:
            print(f"Raw content:\n{text}")
        
    return None

def extract_cheatsheet_bullets(cheatsheet_text, bullet_ids):
    """
    Extract specific bullet points from cheatsheet based on bullet_ids.
    
    Args:
        cheatsheet_text (str): The full cheatsheet text
        bullet_ids (list): List of bullet IDs to extract
    
    Returns:
        str: Formatted cheatsheet content containing only the specified bullets
    """
    if not bullet_ids:
        return "(No bullets used by generator)"
    
    lines = cheatsheet_text.strip().split('\n')
    found_bullets = []
    
    for line in lines:
        if line.strip():  # Skip empty lines
            parsed = parse_cheatsheet_line(line)
            if parsed and parsed['id'] in bullet_ids:
                found_bullets.append({
                    'id': parsed['id'],
                    'content': parsed['content'],
                    'helpful': parsed['helpful'],
                    'harmful': parsed['harmful']
                })
    
    if not found_bullets:
        return "(Generator referenced bullet IDs but none were found in cheatsheet)"
    
    # Format the bullets for reflector input
    formatted_bullets = []
    for bullet in found_bullets:
        formatted_bullets.append(f"[{bullet['id']}] {bullet['content']}")
    
    return '\n'.join(formatted_bullets)

def read_file(file_path: str, mode: Literal["r", "rb"] = "r") -> str | bytes:
    with open(file_path, mode=mode) as file:
        content = file.read()
    return content

class ReflectiveMutationProposer(ProposeNewCandidate):
    """
    Implements current reflective mutation flow:
    - Select candidate via selector
    - Select minibatch via sampler
    - capture_traces_and_eval -> trajectories, subsample_scores
    - skip if all scores==perfect and skip_perfect_score
    - reflection + mutate -> new candidate
    - evaluate new candidate on same minibatch -> new_subsample_scores
    - Return proposal if improved; else None
    """

    def __init__(
        self,
        logger: Any,
        trainset: list[DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler,
        perfect_score: float,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        reflection_lm: LanguageModel | None = None,
    ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.logger = logger
        self.trainset = trainset
        self.adapter = adapter
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.experiment_tracker = experiment_tracker
        self.reflection_lm = reflection_lm
        self.next_global_id = 0
        self.star_prompt = read_file(os.path.join(current_dir, "reflector.txt"))
        self.curator_prompt = read_file(os.path.join(current_dir, "curator.txt"))

    def propose_new_texts(
        self,
        current_cheatsheet: str,
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:

        message_history = reflective_dataset["instruction_prompt"][0]["Message History"]
        world_feedback = reflective_dataset["instruction_prompt"][0]["Feedback"]
        instruction_prompt = reflective_dataset["instruction_prompt"][0]["Instruction Prompt"]

        filled_prompt = (
            self.star_prompt
            .replace("{{test_report}}", world_feedback or "")
            .replace("{{generated_code}}", "See full conversation history below")
            .replace("{{generated_rationale}}", "See full conversation history below")
            .replace("{{spec_or_api_docs}}", "See full conversation history below")
            .replace("{{execution_error}}", "See full conversation history below")
            .replace("{{cheat_sheet}}", current_cheatsheet or "N/A")
            .replace("{{previous_reflection}}", "N/A")
        )

        conversation_history = "\n\n=== FULL CONVERSATION HISTORY ===\n"
        for i, msg in enumerate(message_history):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_history += f"[{i}] {role.upper()}: {content}\n\n"
        
        filled_prompt += conversation_history

        reasoning_text = self.reflection_lm(filled_prompt)

        # Build curator prompt with explicit response format
        content = self.curator_prompt.format(
            initial_generated_code="See full conversation history below",
            final_generated_code="See full conversation history below",
            guidebook=reasoning_text,
            current_cheatsheet=current_cheatsheet,
            question_context=instruction_prompt,
        )

        content += conversation_history

        curator_response = self.reflection_lm(content)

        # Parse JSON (must match explicit response schema: {"reasoning": str, "operations": [...]})
        try:
            operations_info = extract_json_from_text(curator_response, "operations")

            # Strict validation
            if not operations_info:
                raise ValueError("Failed to extract valid JSON from curator response")

            if "reasoning" not in operations_info:
                raise ValueError("JSON missing required 'reasoning' field")
            if "operations" not in operations_info:
                raise ValueError("JSON missing required 'operations' field")

            if not isinstance(operations_info["reasoning"], str):
                raise ValueError("'reasoning' field must be a string")
            if not isinstance(operations_info["operations"], list):
                raise ValueError("'operations' field must be a list")

            # Only ADD operations supported
            for i, op in enumerate(operations_info["operations"]):
                if not isinstance(op, dict):
                    raise ValueError(f"Operation {i} must be a dictionary")
                if "type" not in op:
                    raise ValueError(f"Operation {i} missing required 'type' field")
                if op["type"] != "ADD":
                    raise ValueError(f"Operation {i} has invalid type '{op['type']}'. Only 'ADD' operations are supported in this file")

                required_fields = {"type", "section", "content"}
                missing_fields = required_fields - set(op.keys())
                if missing_fields:
                    raise ValueError(f"ADD operation {i} missing fields: {list(missing_fields)}")

            operations = operations_info["operations"]
            print(f"✅ Curator JSON schema validated successfully: {len(operations)} operations")

            # Apply curated updates
            new_cheatsheet, self.next_global_id = apply_curator_operations(
                current_cheatsheet, operations, self.next_global_id
            )
        except:
            print(f"Passing training point... no JSON object created")
            new_cheatsheet = current_cheatsheet
        return new_cheatsheet

    def propose(self, current_cheatsheet: str) -> str | None:

        for train_datapoint in self.trainset:
            curr_prog = {'instruction_prompt': current_cheatsheet}
            eval_curr = self.adapter.evaluate([train_datapoint], curr_prog, capture_traces=True)
            # 3) Build reflective dataset and propose texts
            reflective_dataset = self.adapter.make_reflective_dataset(curr_prog, eval_curr, [])
            current_cheatsheet = self.propose_new_texts(current_cheatsheet, reflective_dataset, [])

        return current_cheatsheet
