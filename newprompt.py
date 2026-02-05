import time
import pandas as pd
import re
import random

def generate_ai_paragraphs_batch(model, topics_batch, style="neutral", author_name=None, 
                                  paragraphs_per_call=5, word_count_range=(100, 200),
                                  batch_index=0):
    """
    Generates MULTIPLE AI paragraphs in a single API call.
    NOW OPTIMIZED for DIFFERENT topics per batch (natural variety).
    
    Args:
        model: The Gemini model instance
        topics_batch: List of DIFFERENT topics (one paragraph per topic)
        style: "neutral" or "mimicked"
        author_name: For mimicked style
        paragraphs_per_call: Number of paragraphs to generate per API call
        word_count_range: Target word count per paragraph
        batch_index: Used to add variety across batches
    
    Returns:
        List of generated paragraphs (strings)
    """
    min_words, max_words = word_count_range
    num_paragraphs = len(topics_batch)
    
    # Check if topics are different (they should be now!)
    topic_counts = {}
    for topic in topics_batch:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    has_duplicates = any(count > 1 for count in topic_counts.values())
    
    if has_duplicates:
        # This should rarely happen now with the new strategy
        uniqueness_prompt = """
CRITICAL VARIETY REQUIREMENT:
Some topics are repeated in this batch. Each paragraph MUST be completely different:
- Use different opening sentences and structures
- Explore different sub-themes or aspects
- Vary your vocabulary and sentence patterns significantly
- Take different argumentative or narrative approaches
"""
    else:
        # This is the normal case now - all different topics
        uniqueness_prompt = """
VARIETY REQUIREMENT:
Each paragraph addresses a DIFFERENT topic. Ensure each is distinct, well-developed, and standalone.
"""
    
    # Variety angles for additional diversity
    variety_angles = [
        "Explore each topic from a unique perspective or angle.",
        "Use different narrative approaches for each paragraph.",
        "Vary your argumentative and rhetorical strategies.",
        "Balance analytical, descriptive, and reflective tones across paragraphs.",
        "Draw on different intellectual traditions or frameworks for each topic."
    ]
    
    variety_instruction = variety_angles[batch_index % len(variety_angles)]

    if style == "neutral":
        topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics_batch)])
        
        prompt = f"""<start_of_turn>user
You are a professional literary engine capable of producing diverse, creative prose.

TASK:
Write {num_paragraphs} SEPARATE paragraphs, one for each topic below.

TOPICS:
{topics_list}

{uniqueness_prompt}

CONSTRAINTS FOR EACH PARAGRAPH:
- Length: {min_words}-{max_words} words per paragraph
- Style: Informative, modern, and accessible prose
- Prohibited: Do NOT use Victorian, historical, or archaic language
- Variety: {variety_instruction}
- Opening: Start each paragraph with a DIFFERENT type of opening (question, statement, observation, anecdote, etc.)
- Separation: Clearly separate each paragraph with "---PARAGRAPH_BREAK---"

OUTPUT FORMAT:
[Paragraph 1 for topic 1]
---PARAGRAPH_BREAK---
[Paragraph 2 for topic 2]
---PARAGRAPH_BREAK---
[etc.]

CRITICAL REMINDERS:
- Write ONLY the paragraphs with separators
- NO preamble, numbering, titles, or meta-commentary
- Each paragraph must be well-developed and standalone
- Vary sentence structures, vocabulary, and rhetorical approaches
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
            - Varied narrative voices and tones"""
            
            variety_techniques = """
AUSTEN VARIETY TECHNIQUES (use different ones across paragraphs):
- Vary between direct social commentary and character introspection
- Shift levels of irony (subtle, overt, or absent)
- Alternate narrative distances (intimate vs. detached observation)
- Use different sentence rhythms and complexities
"""
        else:  # Dickens
            style_desc = """Charles Dickens's distinctive style:
            - Rich, detailed descriptions with vivid imagery
            - Dramatic, emotionally resonant prose
            - Long, flowing sentences with abundant clauses
            - Social consciousness and moral undertones
            - Use of repetition and parallelism for emphasis
            - Memorable character sketches and atmospheric details"""
            
            variety_techniques = """
DICKENS VARIETY TECHNIQUES (use different ones across paragraphs):
- Vary emotional intensity (restrained vs. dramatic)
- Alternate between character-focused and setting-focused descriptions
- Use different types of imagery (visual, auditory, tactile, olfactory)
- Vary pacing (quick observations vs. lingering descriptions)
- Employ different rhetorical devices
"""

        topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics_batch)])
        
        prompt = f"""<start_of_turn>user
You are an expert literary scholar and mimic specializing in 19th-century prose, capable of producing diverse variations in {author_name}'s voice.

TASK:
Write {num_paragraphs} SEPARATE paragraphs adopting {author_name}'s authorial voice.

TOPICS (one paragraph for each):
{topics_list}

{uniqueness_prompt}

STYLE GUIDE - {author_name.upper()}:
{style_desc}

{variety_techniques}

CONSTRAINTS FOR EACH PARAGRAPH:
- Length: {min_words}-{max_words} words per paragraph
- Authenticity: Match {author_name}'s syntax, vocabulary, and 19th-century sensibilities
- Variety: {variety_instruction}
- Technique: Use DIFFERENT stylistic approaches from {author_name}'s repertoire
- Opening: Vary your opening strategies (scene-setting, character observation, philosophical musing, etc.)
- Separation: Clearly separate each paragraph with "---PARAGRAPH_BREAK---"

OUTPUT FORMAT:
[Paragraph 1 in {author_name}'s style]
---PARAGRAPH_BREAK---
[Paragraph 2 in {author_name}'s style]
---PARAGRAPH_BREAK---
[etc.]

CRITICAL REMINDERS:
- Write ONLY the paragraphs with separators
- NO preamble, numbering, headers, or meta-commentary
- Explore each topic from a unique angle
- Vary sentence structures, vocabulary choices, and rhetorical strategies
- Maintain authenticity to {author_name} while varying stylistic approaches
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
        
        # Clean up any remaining artifacts
        cleaned_paragraphs = []
        for p in paragraphs:
            # Remove potential numbering at start
            p = re.sub(r'^\d+[.)]\s*', '', p.strip())
            # Remove potential topic labels
            p = re.sub(r'^Topic \d+:\s*', '', p, flags=re.IGNORECASE)
            
            if p and len(p.split()) >= 50:
                cleaned_paragraphs.append(p)
        
        # Pad if needed
        while len(cleaned_paragraphs) < num_paragraphs:
            cleaned_paragraphs.append(None)
        
        return cleaned_paragraphs[:num_paragraphs]
    
    except Exception as e:
        print(f"Error generating batch: {e}")
        return [None] * num_paragraphs


def generate_ai_dataset_optimized(model, topics, num_samples_per_topic=50, 
                                   style="neutral", author_name=None, 
                                   batch_size=5):
    """
    NEW STRATEGY: Rotates through topics, generating multiple rounds.
    
    Instead of:
      Topic1 x 50, Topic2 x 50, Topic3 x 50...
    
    We do:
      Round 1: [Topic1, Topic2, Topic3, Topic4, Topic5]
      Round 2: [Topic1, Topic2, Topic3, Topic4, Topic5]
      Round 3: [Topic1, Topic2, Topic3, Topic4, Topic5]
      ... (10 rounds total for 50 samples per topic)
    
    This naturally creates variety because each batch has DIFFERENT topics!
    
    Args:
        model: The Gemini model instance
        topics: List of topic strings
        num_samples_per_topic: How many samples to generate for each topic
        style: "neutral" or "mimicked"
        author_name: For mimicked style (e.g., "Austen", "Dickens")
        batch_size: Number of paragraphs per API call (should match len(topics) or divide evenly)
    
    Returns:
        DataFrame with columns: text, topic, style, author, class, word_count
    """
    samples = []
    num_topics = len(topics)
    
    print(f"TOPIC ROTATION STRATEGY")
    print(f"  - {num_topics} topics")
    print(f"  - {num_samples_per_topic} samples per topic")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Strategy: Rotate through topics (natural variety!)\n")
    
    # Calculate how many complete rounds we need
    num_rounds = num_samples_per_topic
    
    # We'll process in batches, cycling through topics
    total_to_generate = num_topics * num_samples_per_topic
    samples_per_round = num_topics  # One sample per topic per round
    
    # Track how many samples we've generated for each topic
    topic_counters = {topic: 0 for topic in topics}
    
    api_call_count = 0
    round_num = 0
    
    while any(count < num_samples_per_topic for count in topic_counters.values()):
        round_num += 1
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{num_samples_per_topic}")
        print(f"{'='*60}")
        
        # Process topics in batches
        for batch_start in range(0, num_topics, batch_size):
            batch_end = min(batch_start + batch_size, num_topics)
            batch_topics = topics[batch_start:batch_end]
            
            # Only generate for topics that still need samples
            topics_to_generate = []
            topic_indices = []
            
            for idx, topic in enumerate(batch_topics, start=batch_start):
                if topic_counters[topic] < num_samples_per_topic:
                    topics_to_generate.append(topic)
                    topic_indices.append(idx)
            
            if not topics_to_generate:
                continue
            
            api_call_count += 1
            print(f"  API Call {api_call_count}: Generating {len(topics_to_generate)} paragraphs (topics {batch_start+1}-{batch_end})...", end='')
            
            # Generate batch with DIFFERENT topics
            paragraphs = generate_ai_paragraphs_batch(
                model,
                topics_to_generate,
                style=style,
                author_name=author_name,
                paragraphs_per_call=len(topics_to_generate),
                batch_index=api_call_count
            )
            
            # Store results
            for topic, text in zip(topics_to_generate, paragraphs):
                if text:
                    samples.append({
                        'text': text,
                        'topic': topic,
                        'style': style,
                        'author': author_name if author_name else 'AI',
                        'class': f'ai_{style}',
                        'word_count': len(text.split())
                    })
                    topic_counters[topic] += 1
            
            print(f" ✓")
            
            # Rate limiting
            time.sleep(2.0)
        
        # Show progress
        min_count = min(topic_counters.values())
        max_count = max(topic_counters.values())
        print(f"  Progress: {min_count}-{max_count}/{num_samples_per_topic} samples per topic")
        print(f"  Total samples so far: {len(samples)}/{total_to_generate}")
    
    print(f"\n✓ Generation complete!")
    print(f"  - Successfully generated: {len(samples)} samples")
    print(f"  - Target was: {total_to_generate}")
    print(f"  - API calls used: {api_call_count}")
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Check for duplicates
    exact_dupes = df[df.duplicated(subset=['text'], keep=False)]
    print(f"  - Exact duplicates: {len(exact_dupes)}")
    
    if len(exact_dupes) > 0:
        print(f"WARNING: {len(exact_dupes)} duplicates detected!")
    else:
        print(f"  ✓ No exact duplicates found!")
    
    # Near-duplicate check
    first_30_words = df['text'].apply(lambda x: ' '.join(x.split()[:30]))
    near_dupes = df[first_30_words.duplicated(keep=False)]
    print(f"  - Near-duplicates (same first 30 words): {len(near_dupes)}")
    
    return df


def check_dataset_diversity(df, sample_size=5):
    """
    Comprehensive diversity analysis.
    """
    print("\n" + "="*60)
    print("DIVERSITY ANALYSIS")
    print("="*60)
    
    # 1. Exact duplicates
    exact_dupes = df[df.duplicated(subset=['text'], keep=False)]
    print(f"\n1. EXACT DUPLICATES: {len(exact_dupes)}")
    if len(exact_dupes) > 0:
        print(f"     Found {len(exact_dupes)} exact duplicates")
        # Show which topics have duplicates
        dupe_topics = exact_dupes.groupby('topic').size().sort_values(ascending=False)
        print(f"   Topics with duplicates:")
        for topic, count in dupe_topics.head(3).items():
            print(f"     - {topic[:40]}...: {count} duplicates")
    else:
        print(f"   ✓ No exact duplicates!")
    
    # 2. Near-duplicates (same first 50 words)
    first_50_words = df['text'].apply(lambda x: ' '.join(x.split()[:50]))
    near_dupes = df[first_50_words.duplicated(keep=False)]
    print(f"\n2. NEAR-DUPLICATES (same first 50 words): {len(near_dupes)}")
    if len(near_dupes) > 0:
        print(f"     Found {len(near_dupes)} near-duplicates")
    else:
        print(f"   ✓ No near-duplicates!")
    
    # 3. Topic-level variety
    print(f"\n3. TOPIC-LEVEL VARIETY:")
    print(f"   Total topics: {df['topic'].nunique()}")
    print(f"   Checking duplicate rates per topic...")
    
    topic_stats = []
    for topic in df['topic'].unique():
        topic_df = df[df['topic'] == topic]
        topic_dupes = topic_df[topic_df.duplicated(subset=['text'], keep=False)]
        dupe_rate = len(topic_dupes) / len(topic_df) * 100 if len(topic_df) > 0 else 0
        topic_stats.append((topic, len(topic_df), len(topic_dupes), dupe_rate))
    
    # Sort by duplicate rate (highest first)
    topic_stats.sort(key=lambda x: x[3], reverse=True)
    
    print(f"   Top 3 topics with most duplicates:")
    for topic, total, dupes, rate in topic_stats[:3]:
        print(f"     - {topic[:35]}...: {dupes}/{total} ({rate:.1f}%)")
    
    # 4. Opening diversity
    print(f"\n4. OPENING DIVERSITY:")
    first_words = df['text'].apply(lambda x: x.split()[0].lower() if len(x.split()) > 0 else '')
    unique_first_words = len(first_words.unique())
    print(f"   - Unique first words: {unique_first_words}/{len(df)} ({unique_first_words/len(df)*100:.1f}%)")
    print(f"   - Most common first words:")
    for word, count in first_words.value_counts().head(5).items():
        print(f"     '{word}': {count} times")
    
    # 5. Vocabulary diversity
    print(f"\n5. VOCABULARY DIVERSITY:")
    sample_texts = df['text'].head(100)
    all_words = ' '.join(sample_texts).lower().split()
    unique_words = len(set(all_words))
    total_words = len(all_words)
    ttr = unique_words / total_words if total_words > 0 else 0
    print(f"   - Type-Token Ratio (first 100 samples): {ttr:.3f}")
    print(f"   - Unique words: {unique_words}")
    print(f"   - Total words: {total_words}")
    
    # 6. Sample variety check
    print(f"\n6. SAMPLE VARIETY CHECK:")
    if len(df) >= sample_size:
        topic_for_check = df['topic'].iloc[0]
        topic_samples = df[df['topic'] == topic_for_check].head(sample_size)
        
        print(f"   Showing {min(sample_size, len(topic_samples))} samples for: '{topic_for_check[:40]}...'")
        for idx, (i, row) in enumerate(topic_samples.iterrows(), 1):
            first_15 = ' '.join(row['text'].split()[:15])
            print(f"   {idx}. {first_15}...")
    
    # 7. Overall quality score
    print(f"\n7. QUALITY SCORE:")
    exact_score = max(0, 100 - (len(exact_dupes) / len(df) * 100)) if len(df) > 0 else 0
    near_score = max(0, 100 - (len(near_dupes) / len(df) * 100)) if len(df) > 0 else 0
    opening_score = (unique_first_words / len(df) * 100) if len(df) > 0 else 0
    overall = (exact_score + near_score + opening_score) / 3
    
    print(f"   - Exact duplicate score: {exact_score:.1f}/100")
    print(f"   - Near-duplicate score: {near_score:.1f}/100")
    print(f"   - Opening diversity score: {opening_score:.1f}/100")
    print(f"   - OVERALL QUALITY: {overall:.1f}/100")
    
    if overall >= 90:
        print(f"✓ EXCELLENT - Very high diversity!")
    elif overall >= 75:
        print(f"✓ GOOD - Acceptable diversity")
    elif overall >= 60:
        print(f"FAIR - Some variety issues")
    else:
        print(f"POOR - Significant variety problems")
    
    print("\n" + "="*60)


# USAGE EXAMPLE:
"""
import google.generativeai as genai

# Configure API
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-pro')

# Extract topics first (same as before)
topics_list = ['topic1', 'topic2', 'topic3', 'topic4', 'topic5', 
               'topic6', 'topic7', 'topic8', 'topic9', 'topic10']

# Generate Class 2 (AI Neutral)
class2_df = generate_ai_dataset_optimized(
    model=model,
    topics=topics_list,
    num_samples_per_topic=50,
    style="neutral",
    batch_size=5  # Will cycle through 5 topics at a time
)

check_dataset_diversity(class2_df)

# Generate Class 3a (Mimicking Austen)
topics_austen = topics_list[:5]  # First 5 topics

class3a_df = generate_ai_dataset_optimized(
    model=model,
    topics=topics_austen,
    num_samples_per_topic=50,
    style="mimicked",
    author_name="Austen",
    batch_size=5
)

check_dataset_diversity(class3a_df)

# Generate Class 3b (Mimicking Dickens)
topics_dickens = topics_list[5:]  # Last 5 topics

class3b_df = generate_ai_dataset_optimized(
    model=model,
    topics=topics_dickens,
    num_samples_per_topic=50,
    style="mimicked",
    author_name="Dickens",
    batch_size=5
)

check_dataset_diversity(class3b_df)
"""