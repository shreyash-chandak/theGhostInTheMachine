import time
import pandas as pd
import re
import random

def generate_ai_paragraphs_batch(model, topics_batch, style="neutral", author_name=None, 
                                  paragraphs_per_call=5, word_count_range=(100, 200),
                                  batch_index=0):
    """
    Generates MULTIPLE AI paragraphs in a single API call with VARIETY enforcement.
    
    Args:
        topics_batch: List of topics (can be same topic repeated or different topics)
        style: "neutral" or "mimicked"
        author_name: For mimicked style
        paragraphs_per_call: Number of paragraphs to generate per API call
        word_count_range: Target word count per paragraph
        batch_index: Used to add variety across batches (for same topic)
    
    Returns:
        List of generated paragraphs (strings)
    """
    min_words, max_words = word_count_range
    num_paragraphs = len(topics_batch)
    
    # Variety instructions that change per batch
    variety_angles = [
        "Focus on different aspects, perspectives, or time periods for each paragraph.",
        "Explore contrasting viewpoints or different facets of each topic.",
        "Vary your narrative approach: some analytical, some descriptive, some reflective.",
        "Use different argumentative strategies or thematic emphases for each paragraph.",
        "Explore the topics from different angles: historical, contemporary, philosophical, practical."
    ]
    
    variety_instruction = variety_angles[batch_index % len(variety_angles)]
    
    # Add explicit variety prompts based on whether topics are repeated
    topic_counts = {}
    for topic in topics_batch:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    has_duplicates = any(count > 1 for count in topic_counts.values())
    
    if has_duplicates:
        uniqueness_prompt = """
CRITICAL VARIETY REQUIREMENT:
You are writing MULTIPLE paragraphs on the SAME topic. Each paragraph MUST be completely different:
- Use different opening sentences and structures
- Explore different sub-themes or aspects
- Vary your vocabulary and sentence patterns significantly
- Take different argumentative or narrative approaches
- DO NOT repeat the same examples, metaphors, or phrasing
- Imagine you are writing for different audiences or purposes
"""
    else:
        uniqueness_prompt = """
VARIETY REQUIREMENT:
Each paragraph addresses a different topic, so ensure distinct content and style variations.
"""

    if style == "neutral":
        # Build prompt for multiple paragraphs with VARIETY enforcement
        topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics_batch)])
        
        prompt = f"""<start_of_turn>user
You are a professional literary engine capable of producing diverse, creative prose.

TASK:
Write {num_paragraphs} SEPARATE, COMPLETELY UNIQUE paragraphs, one for each topic below.

TOPICS:
{topics_list}

{uniqueness_prompt}

CONSTRAINTS FOR EACH PARAGRAPH:
- Length: {min_words}-{max_words} words per paragraph. This must be strictly enforced. Any paragraph not in the range is INVALID
- Style: Informative, modern, and accessible prose
- Prohibited: Do NOT use Victorian, historical, or archaic language
- Variety: {variety_instruction}
- Opening: Start each paragraph with a DIFFERENT type of opening (question, statement, observation, quote, etc.)
- Separation: Clearly separate each paragraph with "---PARAGRAPH_BREAK---"

OUTPUT FORMAT:
[Unique Paragraph 1 for topic 1]
---PARAGRAPH_BREAK---
[Unique Paragraph 2 for topic 2]
---PARAGRAPH_BREAK---
[etc.]

CRITICAL REMINDERS:
- Write ONLY the paragraphs with separators
- NO preamble, numbering, titles, or meta-commentary
- Each paragraph must be SUBSTANTIALLY DIFFERENT in content, structure, and expression
- Avoid formulaic patterns - be creative and varied
- Make sure that the paragraphs are distinct. Extremely similar paragraphs are also INVALID
<end_of_turn>
<start_of_turn>model
"""

    else:  # mimicked style
        if author_name == "Austen":
            style_desc = """Jane Austen's distinctive style:
            - Ironic, witty social commentary
            - Free indirect discourse (blending narrator and character thoughts)
            - Complex, balanced sentences with multiple clauses
            - Sharp observations about social proprieties and human nature
            - Use of words like 'consequence', 'sensibility', 'propriety', 'establishment'
            - Varied narrative voices and tones across different passages"""
            
            variety_techniques = """
                AUSTEN VARIETY TECHNIQUES (use different ones for each paragraph):
                - Shift between direct social commentary and character introspection
                - Vary the level of irony (subtle vs. overt)
                - Alternate between dialogue-heavy and narrative-heavy passages
                - Use different narrative distances (close vs. distant observation)
                - Vary sentence complexity and rhythm"""
        else:  # Dickens
            style_desc = """Charles Dickens's distinctive style:
            - Rich, detailed descriptions with vivid imagery
            - Dramatic, emotionally resonant prose
            - Long, flowing sentences with abundant clauses
            - Social consciousness and moral undertones
            - Use of repetition and parallelism for emphasis
            - Memorable character sketches and atmospheric details"""
            
            variety_techniques = """
                DICKENS VARIETY TECHNIQUES (use different ones for each paragraph):
                - Vary between emotional intensity levels (restrained vs. dramatic)
                - Alternate between character-focused and setting-focused descriptions
                - Use different types of imagery (visual, auditory, tactile, olfactory)
                - Vary the pacing (quick observations vs. lingering descriptions)
                - Employ different rhetorical devices across paragraphs"""

        topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics_batch)])
        
        prompt = f"""<start_of_turn>user
You are an expert literary scholar and mimic specializing in 19th-century prose, capable of producing diverse variations in {author_name}'s voice.

TASK:
Write {num_paragraphs} SEPARATE, STYLISTICALLY VARIED paragraphs adopting {author_name}'s authorial voice.

TOPICS (one unique paragraph for each):
{topics_list}

{uniqueness_prompt}

STYLE GUIDE - {author_name.upper()}:
{style_desc}

{variety_techniques}

CONSTRAINTS FOR EACH PARAGRAPH:
- Length: {min_words}-{max_words} words per paragraph. This must be strictly enforced. Any paragraph not in the range is INVALID
- Authenticity: Match {author_name}'s syntax, vocabulary, and 19th-century sensibilities
- Variety: {variety_instruction}
- Technique: Use DIFFERENT stylistic approaches from {author_name}'s repertoire for each paragraph
- Opening: Vary your opening strategies (scene-setting, character observation, philosophical musing, etc.)
- Separation: Clearly separate each paragraph with "---PARAGRAPH_BREAK---"

OUTPUT FORMAT:
[Unique Paragraph 1 in {author_name}'s style]
---PARAGRAPH_BREAK---
[Unique Paragraph 2 in {author_name}'s style]
---PARAGRAPH_BREAK---
[etc.]

CRITICAL REMINDERS:
- Write ONLY the paragraphs with separators
- NO preamble, numbering, headers, or meta-commentary
- Each paragraph must explore the topic from a DIFFERENT ANGLE
- Vary sentence structures, vocabulary choices, and rhetorical strategies
- Make sure that the paragraphs are distinct. Extremely similar paragraphs are also INVALID
- Authenticity to {author_name} PLUS substantial variety between paragraphs
<end_of_turn>
<start_of_turn>model
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={'temperature': 0.9, 'top_p': 0.95}
        )
        raw_text = response.text.strip()
        
        # Split by separator
        paragraphs = [p.strip() for p in raw_text.split("---PARAGRAPH_BREAK---")]
        
        # Clean up any remaining artifacts and number markers
        cleaned_paragraphs = []
        for p in paragraphs:
            # Remove potential numbering at start (1., 1), etc.)
            p = re.sub(r'^\d+[.)]\s*', '', p.strip())
            # Remove potential topic labels
            p = re.sub(r'^Topic \d+:\s*', '', p, flags=re.IGNORECASE)
            
            if p and len(p.split()) >= 50:  # Filter very short responses
                cleaned_paragraphs.append(p)
        
        # If we got fewer paragraphs than expected, pad with None
        while len(cleaned_paragraphs) < num_paragraphs:
            cleaned_paragraphs.append(None)
        
        return cleaned_paragraphs[:num_paragraphs]
    
    except Exception as e:
        print(f"Error generating batch: {e}")
        return [None] * num_paragraphs


def generate_ai_dataset_optimized(model, topics, num_samples_per_topic=50, style="neutral", 
                                   author_name=None, batch_size=5, 
                                   use_variety_mode=True):
    """
    Generates a dataset of AI paragraphs using BATCH API calls with VARIETY enforcement.
    
    Args:
        topics: List of topic strings
        num_samples_per_topic: How many samples to generate for each topic
        style: "neutral" or "mimicked"
        author_name: For mimicked style (e.g., "Austen", "Dickens")
        batch_size: Number of paragraphs to generate per API call (default 5)
        use_variety_mode: If True, uses the variety-focused generation approach
    
    Returns:
        DataFrame with columns: text, topic, style, author, class, word_count
    """
    samples = []
    total_to_generate = num_samples_per_topic * len(topics)
    
    if use_variety_mode and batch_size > 1:
        print(f"VARIETY MODE ENABLED")
        print(f" - Using varied batch generation to minimize duplicates")
        print(f" - Explicit uniqueness instructions per batch\n")
    
    total_api_calls = (total_to_generate + batch_size - 1) // batch_size
    
    print(f"Generating {total_to_generate} samples ({num_samples_per_topic} per topic)...")
    print(f"Using batch generation: {batch_size} paragraphs per API call")
    print(f"Total API calls needed: {total_api_calls} (vs {total_to_generate} without batching)")
    print(f"Time savings: ~{((total_to_generate - total_api_calls) * 1.5 / 60):.1f} minutes\n")
    
    # STRATEGY: Instead of just repeating topics, we'll use the variety function
    # for topics that need multiple samples
    if use_variety_mode:
        for topic_idx, topic in enumerate(topics):
            print(f"\n{'='*60}")
            print(f"TOPIC {topic_idx + 1}/{len(topics)}: {topic}")
            print(f"{'='*60}")
            
            # Generate num_samples_per_topic varied samples for this topic
            topic_samples = generate_varied_samples_single_topic(
                model,
                topic=topic,
                num_samples=num_samples_per_topic,
                style=style,
                author_name=author_name,
                batch_size=batch_size
            )
            
            # Add to main samples list
            samples.extend(topic_samples)
    
    else:
        # Original approach (kept as fallback)
        all_topics = []
        for topic in topics:
            all_topics.extend([topic] * num_samples_per_topic)
        
        api_call_count = 0
        for i in range(0, len(all_topics), batch_size):
            batch_topics = all_topics[i:i + batch_size]
            api_call_count += 1
            
            print(f"API Call {api_call_count}/{total_api_calls}: Generating {len(batch_topics)} paragraphs...")
            
            paragraphs = generate_ai_paragraphs_batch(
                model,
                batch_topics, 
                style=style, 
                author_name=author_name,
                paragraphs_per_call=len(batch_topics),
                batch_index=api_call_count
            )
            
            for j, (text, topic) in enumerate(zip(paragraphs, batch_topics)):
                if text:
                    samples.append({
                        'text': text,
                        'topic': topic,
                        'style': style,
                        'author': author_name if author_name else 'AI',
                        'class': f'ai_{style}',
                        'word_count': len(text.split())
                    })
            
            samples_so_far = len(samples)
            print(f"  ✓ Total samples: {samples_so_far}/{total_to_generate}")
            
            if i + batch_size < len(all_topics):
                time.sleep(2.0)
    
    print(f"\n✓ Generation complete!")
    print(f"  - Successfully generated: {len(samples)} samples")
    print(f"  - Failed generations: {total_to_generate - len(samples)}")
    
    # Check for duplicates
    df = pd.DataFrame(samples)
    duplicates = df[df.duplicated(subset=['text'], keep=False)]
    print(f"  - Exact duplicates found: {len(duplicates)}")
    
    if len(duplicates) > 0:
        print(f"  WARNING: {len(duplicates)} duplicates detected!")
        print(f"  Consider using variety mode or reducing batch size")
    
    return df


def generate_varied_samples_single_topic(model, topic, num_samples=50, style="neutral", 
                                         author_name=None, batch_size=5):
    """
    Generates VARIED paragraphs for a SINGLE topic using batch API calls.
    Each paragraph will be DIFFERENT despite being about the same topic.
    
    This is the KEY FUNCTION for avoiding duplicates when generating multiple
    samples for the same topic.
    """
    samples = []
    total_api_calls = (num_samples + batch_size - 1) // batch_size
    
    print(f"  Generating {num_samples} VARIED samples...")
    print(f"  API calls needed: {total_api_calls} (batch size: {batch_size})")
    
    for call_num in range(total_api_calls):
        remaining = num_samples - len(samples)
        current_batch_size = min(batch_size, remaining)
        
        # Create a batch with the SAME topic repeated
        batch_topics = [topic] * current_batch_size
        
        print(f"  API Call {call_num + 1}/{total_api_calls}: Generating {current_batch_size} variations...", end='')
        
        # Generate with variety enforcement
        paragraphs = generate_ai_paragraphs_batch(
            model,
            batch_topics,
            style=style,
            author_name=author_name,
            paragraphs_per_call=current_batch_size,
            batch_index=call_num  # This helps vary prompts across batches
        )
        
        # Add successful generations
        for text in paragraphs:
            if text:
                samples.append({
                    'text': text,
                    'topic': topic,
                    'style': style,
                    'author': f'{author_name}_mimicked' if author_name else 'AI',
                    'class': f'ai_{style}',
                    'word_count': len(text.split())
                })
        
        print(f" ✓ ({len(samples)}/{num_samples} total)")
        
        if len(samples) < num_samples:
            time.sleep(2.0)
    
    return samples


def check_dataset_diversity(df, sample_size=5):
    """
    Analyzes the diversity of generated text to detect duplicates and near-duplicates.
    """
    print("\n" + "="*60)
    print("DIVERSITY ANALYSIS")
    print("="*60)
    
    # Exact duplicates
    exact_dupes = df[df.duplicated(subset=['text'], keep=False)]
    print(f"\n1. EXACT DUPLICATES: {len(exact_dupes)}")
    
    if len(exact_dupes) > 0:
        print(f"   Found {len(exact_dupes)} exact duplicates")
        print(f"   Sample duplicate:")
        print(f"   '{exact_dupes.iloc[0]['text'][:150]}...'")
    
    # Near-duplicates (same first 50 words)
    first_50_words = df['text'].apply(lambda x: ' '.join(x.split()[:50]))
    near_dupes = df[first_50_words.duplicated(keep=False)]
    print(f"\n2. NEAR-DUPLICATES (same first 50 words): {len(near_dupes)}")
    
    # Check variety within topics
    print(f"\n3. TOPIC-LEVEL VARIETY:")
    for topic in df['topic'].unique()[:3]:  # Check first 3 topics
        topic_df = df[df['topic'] == topic]
        topic_dupes = topic_df[topic_df.duplicated(subset=['text'], keep=False)]
        print(f"   - '{topic[:40]}...': {len(topic_dupes)} duplicates out of {len(topic_df)} samples")
    
    # Vocabulary diversity
    print(f"\n4. VOCABULARY DIVERSITY:")
    all_words = ' '.join(df['text'].head(100)).lower().split()
    unique_words = len(set(all_words))
    total_words = len(all_words)
    print(f"   - Type-Token Ratio (first 100 samples): {unique_words/total_words:.3f}")
    
    # Sample variety check
    print(f"\n5. SAMPLE VARIETY CHECK:")
    if len(df) >= sample_size:
        topic_for_check = df['topic'].iloc[0]
        topic_samples = df[df['topic'] == topic_for_check].head(sample_size)
        
        print(f"   Showing {sample_size} samples for topic: '{topic_for_check[:40]}...'")
        for i, row in topic_samples.iterrows():
            first_10 = ' '.join(row['text'].split()[:10])
            print(f"   Sample {i+1}: {first_10}...")
    
    print("\n" + "="*60)


# USAGE EXAMPLE:
"""
# For Class 2 (AI Neutral) - WITH VARIETY MODE
class2_df = generate_ai_dataset_optimized(
    topics=topics_list,
    num_samples_per_topic=50,
    style="neutral",
    batch_size=5,
    use_variety_mode=True  # KEY: This enables variety-focused generation
)

# Check diversity
check_dataset_diversity(class2_df)

# For Class 3a (Mimicking Austen) - WITH VARIETY MODE
class3a_df = generate_ai_dataset_optimized(
    topics=topics_austen,
    num_samples_per_topic=50,
    style="mimicked",
    author_name="Austen",
    batch_size=5,
    use_variety_mode=True  # KEY: Enables variety mode
)

check_dataset_diversity(class3a_df)

# For Class 3b (Mimicking Dickens) - WITH VARIETY MODE  
class3b_df = generate_ai_dataset_optimized(
    topics=topics_dickens,
    num_samples_per_topic=50,
    style="mimicked",
    author_name="Dickens",
    batch_size=5,
    use_variety_mode=True  # KEY: Enables variety mode
)

check_dataset_diversity(class3b_df)
"""