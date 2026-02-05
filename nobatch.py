import time
import pandas as pd
import re
import random
import hashlib

def generate_single_paragraph_varied(model, topic, style="neutral", author_name=None,
                                     word_count_range=(100, 200), context_seed=None):
    """
    Generates a SINGLE paragraph with maximum variety.
    
    Key insight: One topic, one API call = best variety guarantee.
    Each call gets unique context modifiers to prevent patterns.
    
    Args:
        model: Gemini model instance
        topic: Single topic string
        style: "neutral" or "mimicked"
        author_name: For mimicked style
        word_count_range: (min, max) words
        context_seed: Unique seed for this generation (prevents patterns)
    
    Returns:
        Generated paragraph text (string) or None if failed
    """
    min_words, max_words = word_count_range
    
    # Generate unique context modifiers based on seed
    # This prevents the LLM from falling into patterns
    if context_seed is None:
        context_seed = random.randint(0, 1000000)
    
    # Rotate through different "lenses" for each generation
    perspective_lenses = [
        "Explore this topic with fresh eyes, as if discovering it for the first time.",
        "Approach this topic from an unconventional angle or perspective.",
        "Consider both the surface and deeper implications of this topic.",
        "Examine this topic through a critical, analytical lens.",
        "Reflect on this topic with nuance and intellectual depth.",
        "Investigate the subtle complexities of this topic.",
        "Present this topic in a way that challenges conventional thinking.",
        "Illuminate lesser-known aspects or facets of this topic."
    ]
    
    opening_styles = [
        "Begin with a thought-provoking question.",
        "Start with a vivid observation or description.",
        "Open with a bold statement or assertion.",
        "Lead with a relevant anecdote or example.",
        "Commence with a philosophical reflection.",
        "Initiate with a surprising fact or insight.",
        "Start with a comparative or contrastive observation.",
        "Begin with a historical or contextual grounding."
    ]
    
    tonal_variations = [
        "Maintain a contemplative, measured tone.",
        "Adopt a more energetic, engaged voice.",
        "Use a balanced, objective perspective.",
        "Employ a subtly persuasive approach.",
        "Keep the tone intellectually curious.",
        "Use a thoughtful, exploratory manner.",
        "Maintain scholarly neutrality.",
        "Adopt an accessible yet sophisticated voice."
    ]
    
    # Select variations based on context_seed for consistency within same seed
    # but variety across different seeds
    random.seed(context_seed)
    
    perspective = random.choice(perspective_lenses)
    opening = random.choice(opening_styles)
    tone = random.choice(tonal_variations)
    
    # Add random structural variation
    sentence_structures = [
        "Vary sentence length: include both short, punchy sentences and longer, more complex ones.",
        "Use a mix of simple and compound-complex sentences.",
        "Employ rhetorical questions alongside declarative statements.",
        "Balance abstract concepts with concrete examples.",
        "Alternate between active and passive voice where appropriate."
    ]
    structure = random.choice(sentence_structures)
    
    # Reset random seed to ensure true randomness for next call
    random.seed()
    
    if style == "neutral":
        prompt = f"""<start_of_turn>user
You are a professional literary engine producing unique, thoughtful prose.

TOPIC: {topic}

APPROACH:
{perspective}

OPENING STRATEGY:
{opening}

TONE:
{tone}

STRUCTURAL GUIDANCE:
{structure}

CONSTRAINTS:
- Length: {min_words}-{max_words} words, strictly enforeced. Any paragraph not in the range is INVALID
- Style: Modern, accessible, informative prose
- NO Victorian, historical, or archaic language
- Make this paragraph unique and distinctive
- Avoid clichés and formulaic phrasing

Write ONLY the paragraph. No preamble, no title, no meta-commentary.
<end_of_turn>
<start_of_turn>model
"""

    else:  # mimicked style
        if author_name == "Austen":
            style_desc = """Jane Austen's voice:
            - Ironic social commentary with keen observation
            - Free indirect discourse blending narrator and character
            - Balanced, elegant sentences with careful structure
            - Subtle wit and gentle satire
            - 19th-century vocabulary: 'consequence', 'sensibility', 'propriety'"""
            
            austen_variations = [
                "Emphasize social observation and character psychology.",
                "Focus on ironic commentary about manners and society.",
                "Use free indirect discourse to blend perspectives.",
                "Highlight the comedy of social pretensions.",
                "Explore the intersection of propriety and genuine feeling.",
                "Examine social dynamics with sharp but gentle wit.",
                "Reveal character through social interactions and dialogue.",
                "Balance romantic sensibility with rational observation."
            ]
            
            variation_guidance = random.choice(austen_variations)
            
        else:  # Dickens
            style_desc = """Charles Dickens's voice:
            - Vivid, atmospheric descriptions with rich detail
            - Emotionally resonant, dramatic prose
            - Long, flowing sentences with multiple clauses
            - Social consciousness and moral undertones
            - Memorable characterizations and settings
            - Victorian vocabulary with rhetorical flourish"""
            
            dickens_variations = [
                "Emphasize rich, sensory descriptions of setting or character.",
                "Use dramatic, emotionally charged language.",
                "Employ lengthy, complex sentences with abundant clauses.",
                "Focus on the moral or social implications of the topic.",
                "Create vivid characterizations or atmospheric scenes.",
                "Use repetition and parallelism for rhetorical effect.",
                "Blend social criticism with human sympathy.",
                "Paint detailed word-pictures with Victorian richness."
            ]
            
            variation_guidance = random.choice(dickens_variations)
        
        prompt = f"""<start_of_turn>user
You are an expert mimic of {author_name}'s distinctive 19th-century prose style.

TOPIC: {topic}

STYLE GUIDE:
{style_desc}

APPROACH FOR THIS PARAGRAPH:
{variation_guidance}

OPENING STRATEGY:
{opening}

STRUCTURAL GUIDANCE:
{structure}

CONSTRAINTS:
- Length: {min_words}-{max_words} words, strictly enforeced. Any paragraph not in the range is INVALID
- Authenticity: Match {author_name}'s syntax, vocabulary, and sensibilities
- Uniqueness: Make this paragraph distinctive within {author_name}'s style
- NO modern language or contemporary references

Write ONLY the paragraph in {author_name}'s voice. No preamble, no meta-commentary.
<end_of_turn>
<start_of_turn>model
"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.95,  # Higher for more creativity
                'top_p': 0.95,
                'top_k': 40
            }
        )
        
        text = response.text.strip()
        
        # Clean up any artifacts
        text = re.sub(r'^\d+[.)]\s*', '', text)
        text = re.sub(r'^Topic.*?:\s*', '', text, flags=re.IGNORECASE)
        
        # Validate minimum length
        if len(text.split()) < 50:
            return None
            
        return text
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


def generate_ai_dataset_optimal(model, topics, num_samples_per_topic=50,
                                 style="neutral", author_name=None,
                                 show_progress_every=10):
    """
    OPTIMAL STRATEGY: Individual API calls with randomized context.
    
    Why this works best:
    1. No batching = no pattern formation
    2. Each call is independent with unique context
    3. Randomized variations prevent LLM from finding shortcuts
    4. Simple, predictable, reliable
    
    This is slower but GUARANTEES variety.
    
    Args:
        model: Gemini model instance
        topics: List of topics
        num_samples_per_topic: Samples per topic
        style: "neutral" or "mimicked"
        author_name: For mimicked style
        show_progress_every: Show progress every N samples
    
    Returns:
        DataFrame with generated samples
    """
    samples = []
    total_to_generate = len(topics) * num_samples_per_topic
    
    print("="*60)
    print("OPTIMAL GENERATION STRATEGY")
    print("="*60)
    print(f"Topics: {len(topics)}")
    print(f"Samples per topic: {num_samples_per_topic}")
    print(f"Total samples: {total_to_generate}")
    print(f"Strategy: Individual API calls with randomization")
    print(f"Estimated time: ~{total_to_generate * 2.5 / 60:.1f} minutes")
    print("="*60)
    print()
    
    # Shuffle topics to avoid any ordering patterns
    topic_order = []
    for _ in range(num_samples_per_topic):
        shuffled = topics.copy()
        random.shuffle(shuffled)
        topic_order.extend(shuffled)
    
    # Generate samples
    failed_count = 0
    api_call_count = 0
    
    for idx, topic in enumerate(topic_order, 1):
        api_call_count += 1
        
        # Generate unique context seed
        context_seed = hashlib.md5(f"{topic}_{idx}_{random.random()}".encode()).hexdigest()
        context_seed_int = int(context_seed[:8], 16)  # Convert to int
        
        # Generate paragraph
        text = generate_single_paragraph_varied(
            model=model,
            topic=topic,
            style=style,
            author_name=author_name,
            context_seed=context_seed_int
        )
        
        if text:
            samples.append({
                'text': text,
                'topic': topic,
                'style': style,
                'author': author_name if author_name else 'AI',
                'class': f'ai_{style}',
                'word_count': len(text.split())
            })
        else:
            failed_count += 1
        
        # Progress indicator
        if idx % show_progress_every == 0:
            success_rate = ((idx - failed_count) / idx) * 100
            print(f"  Progress: {idx}/{total_to_generate} ({success_rate:.1f}% success rate)")
        
        # Rate limiting (crucial for individual calls)
        time.sleep(2.5)  # Slightly longer for individual calls
    
    print()
    print("="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Successfully generated: {len(samples)}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {len(samples)/total_to_generate*100:.1f}%")
    print(f"API calls: {api_call_count}")
    print("="*60)
    print()
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Immediate quality check
    print("QUALITY CHECK:")
    exact_dupes = df[df.duplicated(subset=['text'], keep=False)]
    print(f"  Exact duplicates: {len(exact_dupes)}")
    
    first_30_words = df['text'].apply(lambda x: ' '.join(x.split()[:30]))
    near_dupes = df[first_30_words.duplicated(keep=False)]
    print(f"  Near-duplicates (first 30 words): {len(near_dupes)}")
    
    if len(exact_dupes) == 0 and len(near_dupes) < 10:
        print("  ✓ EXCELLENT variety!")
    elif len(exact_dupes) < 5 and len(near_dupes) < 20:
        print("  ✓ GOOD variety")
    else:
        print("  ⚠️  Some duplicates detected")
    
    print()
    
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
        print(f"   ⚠️  Found {len(exact_dupes)} exact duplicates")
        dupe_topics = exact_dupes.groupby('topic').size().sort_values(ascending=False)
        print(f"   Topics with most duplicates:")
        for topic, count in dupe_topics.head(3).items():
            print(f"     - {topic[:40]}...: {count}")
    else:
        print(f"   ✓ Perfect! No exact duplicates!")
    
    # 2. Near-duplicates
    first_50_words = df['text'].apply(lambda x: ' '.join(x.split()[:50]))
    near_dupes = df[first_50_words.duplicated(keep=False)]
    print(f"\n2. NEAR-DUPLICATES (same first 50 words): {len(near_dupes)}")
    if len(near_dupes) > 0:
        print(f"   Found {len(near_dupes)} near-duplicates ({len(near_dupes)/len(df)*100:.1f}%)")
    else:
        print(f"   ✓ Perfect! No near-duplicates!")
    
    # 3. Opening diversity
    print(f"\n3. OPENING DIVERSITY:")
    first_words = df['text'].apply(lambda x: x.split()[0].lower() if len(x.split()) > 0 else '')
    unique_first_words = len(first_words.unique())
    print(f"   Unique first words: {unique_first_words}/{len(df)} ({unique_first_words/len(df)*100:.1f}%)")
    
    first_3_words = df['text'].apply(lambda x: ' '.join(x.split()[:3]).lower())
    unique_openings = len(first_3_words.unique())
    print(f"   Unique first 3-word phrases: {unique_openings}/{len(df)} ({unique_openings/len(df)*100:.1f}%)")
    
    # 4. Per-topic analysis
    print(f"\n4. PER-TOPIC ANALYSIS:")
    topic_stats = []
    for topic in df['topic'].unique():
        topic_df = df[df['topic'] == topic]
        topic_dupes = topic_df[topic_df.duplicated(subset=['text'], keep=False)]
        topic_stats.append((topic, len(topic_df), len(topic_dupes)))
    
    topic_stats.sort(key=lambda x: x[2], reverse=True)
    
    print(f"   Topics with duplicates:")
    any_dupes = False
    for topic, total, dupes in topic_stats:
        if dupes > 0:
            print(f"     - {topic[:40]}...: {dupes}/{total}")
            any_dupes = True
    
    if not any_dupes:
        print(f"     ✓ No topics have duplicates!")
    
    # 5. Sample inspection
    print(f"\n5. SAMPLE VARIETY (First Topic):")
    if len(df) >= sample_size:
        first_topic = df['topic'].iloc[0]
        topic_samples = df[df['topic'] == first_topic].head(sample_size)
        
        print(f"   Topic: '{first_topic[:50]}...'")
        print(f"   Showing first 12 words of each sample:")
        for idx, (i, row) in enumerate(topic_samples.iterrows(), 1):
            first_12 = ' '.join(row['text'].split()[:12])
            print(f"     {idx}. {first_12}...")
    
    # 6. Quality score
    print(f"\n6. OVERALL QUALITY SCORE:")
    exact_score = max(0, 100 - (len(exact_dupes) / len(df) * 100)) if len(df) > 0 else 0
    near_score = max(0, 100 - (len(near_dupes) / len(df) * 100)) if len(df) > 0 else 0
    opening_score = (unique_openings / len(df) * 100) if len(df) > 0 else 0
    overall = (exact_score + near_score + opening_score) / 3
    
    print(f"   - Exact duplicate score: {exact_score:.1f}/100")
    print(f"   - Near-duplicate score: {near_score:.1f}/100")
    print(f"   - Opening diversity: {opening_score:.1f}/100")
    print(f"   - OVERALL: {overall:.1f}/100")
    
    if overall >= 95:
        print(f"   ✓✓✓ EXCEPTIONAL - Near-perfect variety!")
    elif overall >= 90:
        print(f"   ✓✓ EXCELLENT - Very high diversity!")
    elif overall >= 80:
        print(f"   ✓ GOOD - Acceptable diversity")
    else:
        print(f"   ⚠️  Needs improvement")
    
    print("\n" + "="*60)
    
    return {
        'exact_duplicates': len(exact_dupes),
        'near_duplicates': len(near_dupes),
        'quality_score': overall
    }


# USAGE EXAMPLE:
"""
import google.generativeai as genai

# Configure API
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-pro')

# Your extracted topics
topics_list = ['topic1', 'topic2', 'topic3', ...]  # 10 topics

# Generate Class 2 (AI Neutral)
class2_df = generate_ai_dataset_optimal(
    model=model,
    topics=topics_list,
    num_samples_per_topic=50,
    style="neutral",
    show_progress_every=25
)

# Check quality
stats = check_dataset_diversity(class2_df)

# Save
class2_df.to_csv('class2_ai_neutral.csv', index=False)


# Generate Class 3a (Austen)
topics_austen = topics_list[:5]

class3a_df = generate_ai_dataset_optimal(
    model=model,
    topics=topics_austen,
    num_samples_per_topic=50,
    style="mimicked",
    author_name="Austen",
    show_progress_every=25
)

check_dataset_diversity(class3a_df)
class3a_df.to_csv('class3a_ai_austen.csv', index=False)


# Generate Class 3b (Dickens)
topics_dickens = topics_list[5:]

class3b_df = generate_ai_dataset_optimal(
    model=model,
    topics=topics_dickens,
    num_samples_per_topic=50,
    style="mimicked",
    author_name="Dickens",
    show_progress_every=25
)

check_dataset_diversity(class3b_df)
class3b_df.to_csv('class3b_ai_dickens.csv', index=False)
"""
