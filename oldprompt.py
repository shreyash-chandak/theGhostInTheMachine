def generate_ai_paragraphs_batch(topics_batch, style="neutral", author_name=None, 
                                  paragraphs_per_call=5, word_count_range=(100, 200)):
    """
    Generates MULTIPLE AI paragraphs in a single API call for efficiency.
    
    Args:
        topics_batch: List of topics (can be same topic repeated or different topics)
        style: "neutral" or "mimicked"
        author_name: For mimicked style
        paragraphs_per_call: Number of paragraphs to generate per API call
        word_count_range: Target word count per paragraph
    
    Returns:
        List of generated paragraphs (strings)
    """
    min_words, max_words = word_count_range
    num_paragraphs = len(topics_batch)

    if style == "neutral":
        # Build prompt for multiple paragraphs
        topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics_batch)])
        
        prompt = f"""<start_of_turn>user
You are a professional literary engine.

TASK:
Write {num_paragraphs} SEPARATE thoughtful paragraphs, one for each topic below.

TOPICS:
{topics_list}

CONSTRAINTS FOR EACH PARAGRAPH:
- Length: {min_words}-{max_words} words per paragraph
- Style: Informative, modern, and accessible
- Prohibited: Do NOT use Victorian, historical, or archaic language
- Separation: Clearly separate each paragraph with "---PARAGRAPH_BREAK---"
- Output format:
  [Paragraph 1 for topic 1]
  ---PARAGRAPH_BREAK---
  [Paragraph 2 for topic 2]
  ---PARAGRAPH_BREAK---
  [etc.]

IMPORTANT: 
- Write ONLY the paragraphs with the separator between them
- No preamble, no numbering, no titles, no meta-commentary
- Each paragraph should be complete and standalone
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
            - Use of words like 'consequence', 'sensibility', 'propriety'"""
        else:  # Dickens
            style_desc = """Charles Dickens's distinctive style:
            - Rich, detailed descriptions with vivid imagery
            - Dramatic, emotionally resonant prose
            - Long, flowing sentences with abundant clauses
            - Social consciousness and moral undertones
            - Use of repetition and parallelism for emphasis"""

        topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics_batch)])
        
        prompt = f"""<start_of_turn>user
You are an expert literary scholar and mimic specializing in 19th-century prose.

TASK:
Write {num_paragraphs} SEPARATE paragraphs by flawlessly adopting the authorial voice of {author_name}.

TOPICS (one paragraph for each):
{topics_list}

STYLE GUIDE:
{style_desc}

CONSTRAINTS FOR EACH PARAGRAPH:
- Length: {min_words}-{max_words} words per paragraph
- Syntax: Match the specific sentence structures and rhythmic patterns of {author_name}
- Vocabulary: Use authentic 19th-century terminology appropriate for the author
- Separation: Clearly separate each paragraph with "---PARAGRAPH_BREAK---"
- Output format:
  [Paragraph 1 in {author_name}'s style]
  ---PARAGRAPH_BREAK---
  [Paragraph 2 in {author_name}'s style]
  ---PARAGRAPH_BREAK---
  [etc.]

IMPORTANT:
- Write ONLY the paragraphs with the separator between them
- No preamble, no numbering, no headers, no meta-commentary
- Each paragraph should be complete and standalone
<end_of_turn>
<start_of_turn>model
"""

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Split by separator
        paragraphs = [p.strip() for p in raw_text.split("---PARAGRAPH_BREAK---")]
        
        # Clean up any remaining artifacts
        paragraphs = [p for p in paragraphs if p and len(p.split()) >= 50]  # Filter very short responses
        
        # If we got fewer paragraphs than expected, pad with None
        while len(paragraphs) < num_paragraphs:
            paragraphs.append(None)
        
        return paragraphs[:num_paragraphs]  # Return exactly the number requested
    
    except Exception as e:
        print(f"Error generating batch: {e}")
        return [None] * num_paragraphs


def generate_ai_dataset_optimized(topics, num_samples_per_topic=50, style="neutral", 
                                   author_name=None, batch_size=5):
    """
    Generates a dataset of AI paragraphs using BATCH API calls for efficiency.
    
    Args:
        topics: List of topic strings
        num_samples_per_topic: How many samples to generate for each topic
        style: "neutral" or "mimicked"
        author_name: For mimicked style (e.g., "Austen", "Dickens")
        batch_size: Number of paragraphs to generate per API call (default 5)
    
    Returns:
        DataFrame with columns: text, topic, style, author, class, word_count
    """
    samples = []
    total_to_generate = num_samples_per_topic * len(topics)
    total_api_calls = (total_to_generate + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Generating {total_to_generate} samples ({num_samples_per_topic} per topic)...")
    print(f"Using batch generation: {batch_size} paragraphs per API call")
    print(f"Total API calls needed: {total_api_calls} (vs {total_to_generate} without batching)")
    print(f"Time savings: ~{((total_to_generate - total_api_calls) * 1.5 / 60):.1f} minutes\n")
    
    # Create a list of all topic assignments
    # e.g., if topics = ['topic1', 'topic2'] and num_samples_per_topic = 50
    # we get ['topic1']*50 + ['topic2']*50
    all_topics = []
    for topic in topics:
        all_topics.extend([topic] * num_samples_per_topic)
    
    # Process in batches
    api_call_count = 0
    for i in range(0, len(all_topics), batch_size):
        batch_topics = all_topics[i:i + batch_size]
        api_call_count += 1
        
        print(f"API Call {api_call_count}/{total_api_calls}: Generating {len(batch_topics)} paragraphs...")
        
        # Generate batch
        paragraphs = generate_ai_paragraphs_batch(
            batch_topics, 
            style=style, 
            author_name=author_name,
            paragraphs_per_call=len(batch_topics)
        )
        
        # Add successful generations to samples
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
        
        # Progress update
        samples_so_far = len(samples)
        print(f"  ✓ Total samples generated so far: {samples_so_far}/{total_to_generate}")
        
        # Rate limiting (still important, but now for batches)
        if i + batch_size < len(all_topics):  # Don't sleep after last batch
            time.sleep(2.0)  # Slightly longer sleep since each call is doing more work
    
    print(f"\n✓ Batch generation complete!")
    print(f"  - Successfully generated: {len(samples)} samples")
    print(f"  - Failed generations: {total_to_generate - len(samples)}")
    
    return pd.DataFrame(samples)


# Alternative: Generate multiple samples for a SINGLE topic (useful for variety)
def generate_varied_samples_single_topic(topic, num_samples=50, style="neutral", 
                                         author_name=None, batch_size=5):
    """
    Generates multiple varied paragraphs for a SINGLE topic using batch API calls.
    Each paragraph should be different despite being about the same topic.
    
    This is useful when you want variety within the same topic.
    """
    samples = []
    total_api_calls = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} varied samples for topic: '{topic}'")
    print(f"Using {total_api_calls} API calls (batch size: {batch_size})")
    
    for call_num in range(total_api_calls):
        remaining = num_samples - len(samples)
        current_batch_size = min(batch_size, remaining)
        
        # Repeat the same topic for variety
        batch_topics = [topic] * current_batch_size
        
        print(f"API Call {call_num + 1}/{total_api_calls}: Generating {current_batch_size} variations...")
        
        paragraphs = generate_ai_paragraphs_batch(
            batch_topics,
            style=style,
            author_name=author_name,
            paragraphs_per_call=current_batch_size
        )
        
        for text in paragraphs:
            if text:
                samples.append({
                    'text': text,
                    'topic': topic,
                    'style': style,
                    'author': author_name if author_name else 'AI',
                    'class': f'ai_{style}',
                    'word_count': len(text.split())
                })
        
        if len(samples) < num_samples:
            time.sleep(2.0)
    
    return pd.DataFrame(samples)