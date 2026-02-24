
import re
import html
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

class BaseCleaner(ABC):
    """Base interface for all text cleaners"""
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Override to compile regex patterns once for performance"""
        pass
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean text and return result"""
        pass
    
    def _safe_clean(self, text: str) -> str:
        """Wrapper with null/empty checks"""
        if not text or not isinstance(text, str):
            return ""
        return self.clean(text)


class CleanerGroup(BaseCleaner):
    """Groups multiple cleaners together"""
    
    def __init__(self, cleaners: list):
        self.cleaners = cleaners
        super().__init__()
    
    def clean(self, text: str) -> str:
        for cleaner in self.cleaners:
            text = cleaner._safe_clean(text)
        return text
        
class UniversalEncodingFixer(BaseCleaner):
    """Fix ALL encoding issues - universal solution"""
    
    def clean(self, text: str) -> str:
        if not text:
            return text
        
        import html
        from ftfy import fix_text
        
        # Method 1: ftfy - catches 99% of issues
        try:
            text = fix_text(text)
        except:
            pass
        
        # Method 2: HTML entity decoding (multiple passes)
        for _ in range(3):
            text = html.unescape(text)
        
        # Method 3: Fix UTF-8 mojibake by re-encoding (only if issues remain)
        try:
            if 'â€' in text or ('Ã' in text and any(c in text for c in 'áéíóúñçü')):
                text = text.encode('latin-1', errors='ignore').decode('utf-8', errors='replace')
                text = text.replace('�', '')
        except:
            pass
        
        # Method 4: Remove ONLY truly non-printable chars (not accented letters!)
        # Keep: letters (including accented), digits, punctuation, spaces
        cleaned_chars = []
        for char in text:
            # Keep if printable OR common whitespace OR accented characters
            if char.isprintable() or char in '\n\t ' or ord(char) > 127:
                cleaned_chars.append(char)
        text = ''.join(cleaned_chars)
        
        # Method 5: Final HTML decode pass
        text = html.unescape(text)
        
        return text
class RegexCleaner(BaseCleaner):
    """Base class for regex-based cleaning"""
    
    def __init__(self, patterns: list, replacement: str = ''):
        self.patterns = patterns
        self.replacement = replacement
        super().__init__()
    
    def _compile_patterns(self):
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.patterns
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.compiled_patterns:
            text = pattern.sub(self.replacement, text)
        return text
class TextCleaningPipeline:
    """Main pipeline - routes text through appropriate cleaners by source"""
    
    def __init__(self):
        self.cleaners = self._build_cleaners()
    
    def _build_cleaners(self):
        """Build all cleaner groups - will add cleaners in next steps"""
        return {
            'Wikipedia': [],
            'NewsAPI': [],
            'Reddit': [],
            'Twitter': [],
        }
    
    def clean(self, text: str, source: str) -> str:
        """Clean text based on source type"""
        if not text or not isinstance(text, str):
            return ""
        
        if source not in self.cleaners:
            source = 'Twitter'  # Default fallback
        
        cleaned_text = text
        for cleaner in self.cleaners[source]:
            cleaned_text = cleaner._safe_clean(cleaned_text)
        
        return cleaned_text
    
    def clean_dataframe(self, df: 'pd.DataFrame', text_col: str = 'text', 
                       source_col: str = 'source') -> 'pd.DataFrame':
        """Apply cleaning to entire dataframe"""
        df['cleaned_text'] = df.apply(
            lambda row: self.clean(row[text_col], row[source_col]),
            axis=1
        )
        return df


class WhitespaceCleaner(BaseCleaner):
    """Final whitespace normalization - runs on all sources"""
    
    def _compile_patterns(self):
        self.patterns = [
            (re.compile(r'[\t\n\r]+'), ' '),           # tabs/newlines to space
            (re.compile(r'\s+'), ' '),                  # collapse multiple spaces
            (re.compile(r'\s+([.,!?;])'), r'\1'),      # fix spacing before punctuation
            (re.compile(r'([.,!?;])\1+'), r'\1'),      # remove repeated punctuation
            (re.compile(r'\s*[,;]\s*[,;]+'), ','),     # collapse repeated commas/semicolons
        ]
    
    def clean(self, text: str) -> str:
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        return text.strip()

# Initialize pipeline
#pipeline = setup_cleaning_pipeline()

# Clean single text
#text = "Check   this  out!!!  Amazing stuff.  "
#cleaned = pipeline.clean(text, source='Twitter')
#print(f"'{cleaned}'")  # 'Check this out! Amazing stuff.'

# Clean your master_context_df
#cleaned_df = pipeline.clean_dataframe(master_context_df)
#print(cleaned_df[['source', 'text', 'cleaned_text']].head())

class URLCleaner(BaseCleaner):
    """Remove all URL types including short links and embedded URLs"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'https?://[^\s]+', re.IGNORECASE),
            re.compile(r'www\.[^\s]+', re.IGNORECASE),
            re.compile(r't\.co/[^\s]+', re.IGNORECASE),
            re.compile(r'\b(?:tinyurl|ow\.ly|is\.gd|buff\.ly|goo\.gl|short\.link)\.(?:com|ly|gd)(?:/[^\s]+)?', re.IGNORECASE),
            re.compile(r'\bbit\.ly(?:/[^\s]+)?', re.IGNORECASE),
            re.compile(r'\b[a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|io|co|ai)/[^\s]+', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class ImageURLCleaner(BaseCleaner):
    """Remove image-specific URLs and platforms"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'https?://[^\s]*\.(?:jpg|jpeg|png|gif|webp|svg|bmp|ico|tiff|heic)[^\s]*', re.IGNORECASE),
            re.compile(r'\b[a-zA-Z0-9-]+\.[a-z]{2,}/[^\s]*\.(?:jpg|jpeg|png|gif|webp)[^\s]*', re.IGNORECASE),
            re.compile(r'pic\.twitter\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'i\.redd\.it/[^\s]+', re.IGNORECASE),
            re.compile(r'(?:i\.)?imgur\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'(?:media\.)?giphy\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'gfycat\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'tenor\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'pbs\.twimg\.com/[^\s]+', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class VideoURLCleaner(BaseCleaner):
    """Remove video URLs and platforms"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'https?://[^\s]*\.(?:mp4|mov|avi|webm|mkv|flv|wmv)[^\s]*', re.IGNORECASE),
            re.compile(r'(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s]+', re.IGNORECASE),
            re.compile(r'vimeo\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'tiktok\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'v\.redd\.it/[^\s]+', re.IGNORECASE),
            re.compile(r'streamable\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'(?:clips\.)?twitch\.tv/[^\s]+', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class AudioURLCleaner(BaseCleaner):
    """Remove audio/podcast URLs"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'https?://[^\s]*\.(?:mp3|wav|flac|aac|ogg|m4a)[^\s]*', re.IGNORECASE),
            re.compile(r'(?:open\.)?spotify\.com/(?:episode|show)/[^\s]+', re.IGNORECASE),
            re.compile(r'soundcloud\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'anchor\.fm/[^\s]+', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class DocumentURLCleaner(BaseCleaner):
    """Remove document/file URLs"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'https?://[^\s]*\.(?:pdf|doc|docx|xls|xlsx|ppt|pptx)[^\s]*', re.IGNORECASE),
            re.compile(r'(?:docs|drive)\.google\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'dropbox\.com/[^\s]+', re.IGNORECASE),
            re.compile(r'onedrive\.live\.com/[^\s]+', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text
        
class DoubleHashCleaner(BaseCleaner):
    """Remove multiple consecutive # symbols (##, ###)"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'#{2,}\s*')  # Match 2+ hashes with optional space
    
    def clean(self, text: str) -> str:
        return self.pattern.sub('', text)


class RedditTableCleaner(BaseCleaner):    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\|'),  # Remove all pipe symbols
            
            # Match table alignment patterns
            re.compile(r'(?::\s*-+\s*:?\s*){1,}'),  # Matches :- -: :--: etc (any count)
            
            # Match standalone separator lines
            re.compile(r'-{3,}'),  # 3+ consecutive dashes
            
            # Catch any remaining colon-dash combinations
            re.compile(r'[:\-]{3,}'),  # 3+ mixed colons and dashes
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub(' ', text)  # Replace with space
        return text
        
class MarkdownHashtagCleaner(BaseCleaner):
    """Handle #[text](url) hybrid pattern"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'#\[([^\]]+)\]\([^\)]+\)')
    
    def clean(self, text: str) -> str:
        return self.pattern.sub(r'#\1', text)

class MarkdownLinkCleaner(BaseCleaner):
    """Remove markdown and HTML links, keep visible text"""
    
    def _compile_patterns(self):
        self.markdown_pattern = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
        self.html_pattern = re.compile(r'<a\s+(?:[^>]*?\s+)?href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', re.IGNORECASE)
        self.bare_brackets = re.compile(r'\[([^\]]+)\]')
    
    def clean(self, text: str) -> str:
        text = self.markdown_pattern.sub(r'\1', text)
        text = self.html_pattern.sub(r'\2', text)
        text = self.bare_brackets.sub(r'\1', text)
        return text


def add_url_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add URL cleaners"""
    
    markdown_hashtag_cleaner = MarkdownHashtagCleaner()  # NEW
    url_cleaner = URLCleaner()
    image_cleaner = ImageURLCleaner()
    video_cleaner = VideoURLCleaner()
    audio_cleaner = AudioURLCleaner()
    doc_cleaner = DocumentURLCleaner()
    markdown_cleaner = MarkdownLinkCleaner()
    
    social_cleaners = [
        markdown_hashtag_cleaner,  # NEW - FIRST
        markdown_cleaner,
        url_cleaner,
        image_cleaner,
        video_cleaner,
        audio_cleaner,
        doc_cleaner,
    ]
    
    news_cleaners = [
        markdown_hashtag_cleaner,  # NEW - FIRST
        markdown_cleaner,
        url_cleaner,
        image_cleaner,
    ]
    
    for cleaner in reversed(social_cleaners):
        pipeline.cleaners['Twitter'].insert(0, cleaner)
        pipeline.cleaners['Reddit'].insert(0, cleaner)
    
    for cleaner in reversed(news_cleaners):
        pipeline.cleaners['NewsAPI'].insert(0, cleaner)
    
    pipeline.cleaners['Wikipedia'].insert(0, url_cleaner)
    pipeline.cleaners['Wikipedia'].insert(0, markdown_cleaner)
    pipeline.cleaners['Wikipedia'].insert(0, markdown_hashtag_cleaner)  # NEW

class MentionCleaner(BaseCleaner):
    """Remove @ symbol but keep username"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'@(\w+)', re.IGNORECASE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub(r'\1', text)

class BotFooterCleaner(BaseCleaner):
    """Remove bot-generated footers"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'Sent from my iPhone', re.IGNORECASE),
            re.compile(r'Posted via.*', re.IGNORECASE),
            re.compile(r'Sent from.*mobile', re.IGNORECASE),
            re.compile(r'Sent from.*app', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class RetweetCleaner(BaseCleaner):
    """Remove RT prefix - works with or without @"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'^\s*RT\s+@?\w+:\s*', re.IGNORECASE),  # Added @? to make @ optional
            re.compile(r'^\s*Retweeted\s+@?\w+:\s*', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text

class ReplyContextCleaner(BaseCleaner):
    """Remove 'Replying to @user' context"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'^\s*Replying to\s+@\w+(?:\s+and\s+@\w+)*\s*', re.IGNORECASE),
            re.compile(r'^\s*In reply to\s+@\w+\s*', re.IGNORECASE),
            re.compile(r'↩️\s*@\w+', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class ThreadMarkerCleaner(BaseCleaner):
    """Remove thread indicators"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'^\s*\d+/\d+\s+', re.IGNORECASE),  # Changed \s* to \s+ (require space)
            re.compile(r'\(\d+/\d+\)', re.IGNORECASE),
            re.compile(r'^\s*Part\s+\d+/\d+:?\s*', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text

class SocialActionCleaner(BaseCleaner):
    """Remove social media action buttons and engagement text"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\b(share|like|comment|reply|retweet|quote tweet)\b', re.IGNORECASE),
            re.compile(r'\b(follow|subscribe|unfollow|unsubscribe)\b', re.IGNORECASE),
            re.compile(r'\b(upvote|downvote|vote)\b', re.IGNORECASE),
            re.compile(r'\b(bookmark|save|flag)\b', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class EngagementCountCleaner(BaseCleaner):
    """Remove view/like/share counts"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\b\d+\.?\d*[KMB]?\s*(likes?|views?|shares?|retweets?|comments?)\b', re.IGNORECASE),
            re.compile(r'\b(Viewed|Watched)\s+\d+\.?\d*[KMB]?\s*times?\b', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


def add_interaction_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add user interaction cleaners"""
    
    mention_cleaner = MentionCleaner()
    bot_footer_cleaner = BotFooterCleaner()
    retweet_cleaner = RetweetCleaner()
    reply_cleaner = ReplyContextCleaner()
    thread_cleaner = ThreadMarkerCleaner()
    social_action_cleaner = SocialActionCleaner()
    engagement_cleaner = EngagementCountCleaner()
    
    # Twitter gets all interaction cleaners
    twitter_cleaners = [
        retweet_cleaner,
        reply_cleaner,
        thread_cleaner,
        mention_cleaner,
        bot_footer_cleaner,
        social_action_cleaner,
        engagement_cleaner,
    ]
    
    # Reddit gets most
    reddit_cleaners = [
        retweet_cleaner,  # ADD THIS - Reddit users share tweets
        mention_cleaner,
        social_action_cleaner,
        engagement_cleaner,
    ]
    
    # Add in reverse order
    for cleaner in reversed(twitter_cleaners):
        pipeline.cleaners['Twitter'].insert(0, cleaner)
    
    for cleaner in reversed(reddit_cleaners):
        pipeline.cleaners['Reddit'].insert(0, cleaner)
    
    # NewsAPI/Wikipedia need basic cleaners
    pipeline.cleaners['NewsAPI'].insert(0, retweet_cleaner)  # ADD
    pipeline.cleaners['NewsAPI'].insert(0, mention_cleaner)
    pipeline.cleaners['NewsAPI'].insert(0, engagement_cleaner)
    
    pipeline.cleaners['Wikipedia'].insert(0, retweet_cleaner)  # ADD
    pipeline.cleaners['Wikipedia'].insert(0, mention_cleaner)
    pipeline.cleaners['Wikipedia'].insert(0, engagement_cleaner)

class HashtagSpamCleaner(BaseCleaner):
    """Remove clusters of 3+ hashtags ONLY at the end of posts with other content"""
    
    def _compile_patterns(self):
        # Match 3+ hashtags at END of text (after real content)
        self.end_cluster = re.compile(r'\s+(?:#\w+\s*){3,}$', re.IGNORECASE)
    
    def clean(self, text: str) -> str:
        # Only remove if there's text before the hashtags
        return self.end_cluster.sub('', text)


class HashtagCleaner(BaseCleaner):
    """Remove # symbol but keep hashtag text"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'#(\w+)', re.IGNORECASE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub(r'\1', text)


class FinalMentionCleaner(BaseCleaner):
    """Final pass to catch any remaining @mentions"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'@(\w+)', re.IGNORECASE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub(r'\1', text)


def add_hashtag_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add hashtag cleaners - run AFTER markdown links are cleaned"""
    
    hashtag_spam_cleaner = HashtagSpamCleaner()
    hashtag_cleaner = HashtagCleaner()
    final_mention_cleaner = FinalMentionCleaner()
    
    cleaners_in_order = [
        hashtag_spam_cleaner,
        hashtag_cleaner,
        final_mention_cleaner,
    ]
    
    # CRITICAL: Don't insert at position 0 (that puts them BEFORE markdown)
    # Instead, insert at position AFTER url cleaners
    for source in ['Twitter', 'Reddit']:
        # Twitter/Reddit have 6 URL cleaners, insert after them
        insert_pos = 6
        for i, cleaner in enumerate(cleaners_in_order):
            pipeline.cleaners[source].insert(insert_pos + i, cleaner)
    
    # NewsAPI/Wikipedia have fewer URL cleaners
    for source in ['NewsAPI', 'Wikipedia']:
        # These have 2-3 cleaners, insert after them
        insert_pos = len([c for c in pipeline.cleaners[source] if 'Cleaner' in c.__class__.__name__ and 'URL' in c.__class__.__name__ or 'Markdown' in c.__class__.__name__])
        for i, cleaner in enumerate(cleaners_in_order):
            pipeline.cleaners[source].insert(insert_pos + i, cleaner)

class EmojiCleaner(BaseCleaner):
    """Convert emojis to text descriptions for sentiment analysis"""
    
    def _compile_patterns(self):
        # Common emojis with their text equivalents
        self.emoji_map = {
            # Reactions
            '👍': ' thumbs up ',
            '👎': ' thumbs down ',
            '❤️': ' heart ',
            '😂': ' laughing ',
            '😢': ' crying ',
            '😮': ' surprised ',
            '😡': ' angry ',
            '😁': ' grinning ',
            '😎': ' cool ',
            '🔥': ' fire ',
            '✨': ' sparkles ',
            '🎉': ' celebration ',
            '🎊': ' party ',
            
            # Medals
            '🥇': ' gold medal ',
            '🥈': ' silver medal ',
            '🥉': ' bronze medal ',
            '🏆': ' trophy ',
            
            # Flags - just indicate it's a flag
            '🇨🇦': ' Canada ',
            '🇺🇸': ' USA ',
            '🇬🇧': ' UK ',
            # Add more country flags as needed
            
            # Pointing
            '👇': ' pointing down ',
            '👆': ' pointing up ',
            '👉': ' pointing right ',
            '👈': ' pointing left ',
            
            # Thread/Media markers
            '🧵': ' thread ',
            '🎥': ' video ',
            '📸': ' photo ',
            '🎵': ' music ',
            '🎧': ' audio ',
            '🔴': ' live ',
            '📈': ' trending ',
            '📍': ' location ',
            
            # Status indicators (remove these)
            '🟡': '',
            '🔵': '',
            '⚪': '',
            '🟢': '',
            
            # Common social media emojis
            '💯': ' hundred ',
            '🙏': ' praying ',
            '💪': ' strong ',
            '⚡': ' lightning ',
            '🌟': ' star ',
        }
        
        # Compile pattern to match any remaining emoji not in our map
        self.emoji_pattern = re.compile(
            r'[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]|'
            r'[\U0001F1E0-\U0001F1FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]',
            re.UNICODE
        )
    
    def clean(self, text: str) -> str:
        # First replace known emojis with text
        for emoji, description in self.emoji_map.items():
            text = text.replace(emoji, description)
        
        # Remove any remaining emojis not in our map
        text = self.emoji_pattern.sub('', text)
        
        return text


class EmojiRemover(BaseCleaner):
    """Remove all emojis completely (for RAG/fact-checking)"""
    
    def _compile_patterns(self):
        # Pattern to match all emojis
        self.emoji_pattern = re.compile(
            r'[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]|'
            r'[\U0001F1E0-\U0001F1FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]',
            re.UNICODE
        )
    
    def clean(self, text: str) -> str:
        return self.emoji_pattern.sub('', text)


def add_emoji_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add emoji handlers - converts to text for social media"""
    
    emoji_cleaner = EmojiCleaner()  # Converts to text (keeps sentiment)
    
    # Social media keeps emoji sentiment as text
    pipeline.cleaners['Twitter'].insert(0, emoji_cleaner)
    pipeline.cleaners['Reddit'].insert(0, emoji_cleaner)
    
    # Factual sources remove emojis (shouldn't have any anyway)
    emoji_remover = EmojiRemover()
    pipeline.cleaners['Wikipedia'].insert(0, emoji_remover)
    pipeline.cleaners['NewsAPI'].insert(0, emoji_remover)

class MarkdownHeaderCleaner(BaseCleaner):
    """Remove markdown headers (# Header)"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'^#+\s+', re.MULTILINE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub('', text)


class BoldItalicCleaner(BaseCleaner):
    """Remove bold/italic markers, keep text"""
    
    def _compile_patterns(self):
        self.bold_pattern = re.compile(r'\*\*([^\*]+)\*\*')
        self.italic_pattern = re.compile(r'\*([^\*]+)\*')
        self.bold_underscore = re.compile(r'__([^_]+)__')
        self.italic_underscore = re.compile(r'_([^_]+)_')
    
    def clean(self, text: str) -> str:
        text = self.bold_pattern.sub(r'\1', text)
        text = self.bold_underscore.sub(r'\1', text)
        text = self.italic_pattern.sub(r'\1', text)
        text = self.italic_underscore.sub(r'\1', text)
        return text


class QuoteMarkerCleaner(BaseCleaner):
    """Remove quote markers (>)"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'^>\s+', re.MULTILINE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub('', text)


class BulletPointCleaner(BaseCleaner):
    """Remove bullet point markers"""
    
    def _compile_patterns(self):
        self.bullet_pattern = re.compile(r'^\*\s+', re.MULTILINE)
        self.dash_pattern = re.compile(r'^-\s+', re.MULTILINE)
        self.numbered_pattern = re.compile(r'^\d+\.\s+', re.MULTILINE)
    
    def clean(self, text: str) -> str:
        text = self.bullet_pattern.sub('', text)
        text = self.dash_pattern.sub('', text)
        text = self.numbered_pattern.sub('', text)
        return text


class CodeBlockCleaner(BaseCleaner):
    """Remove code blocks"""
    
    def _compile_patterns(self):
        self.fenced_code = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.inline_code = re.compile(r'`([^`]+)`')
    
    def clean(self, text: str) -> str:
        text = self.fenced_code.sub('', text)
        text = self.inline_code.sub(r'\1', text)
        return text

class RedditFlairCleaner(BaseCleaner):
    """Remove Reddit post flairs [Serious], [OC], etc."""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'\[(?:Serious|Discussion|OC|NSFW|Spoiler|META|News|Megathread)\]', re.IGNORECASE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub('', text)


class RedditPostLabelCleaner(BaseCleaner):
    """Remove Reddit post labels like 'Competition:', 'Date:', etc."""
    
    def _compile_patterns(self):
        labels = [
            'Competition', 'Date', 'Venue', 'Referee', 'Score',
            'Pics or Text', 'Match Events', 'Basic info',
            'Full club name', 'Subreddit', 'Location', 'Stadium', 'Head Coach'
        ]
        pattern = r'\b(?:' + '|'.join(labels) + r'):\s*'
        self.pattern = re.compile(pattern, re.IGNORECASE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub('', text)


class MatchTimestampCleaner(BaseCleaner):
    """Remove match event timestamps like **07'** ⚽"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r"\*?\*?\d+'\*?\*?\s*[⚽🔄📍⚠️🟨🟥]*\s*", re.UNICODE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub('', text)


class RedditArtifactCleaner(BaseCleaner):
    """Remove Reddit artifacts [removed], [deleted], etc."""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\[removed\]', re.IGNORECASE),
            re.compile(r'\[deleted\]', re.IGNORECASE),
            re.compile(r'\[View Poll\]', re.IGNORECASE),
            re.compile(r'\[See image\]', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class EditMarkerCleaner(BaseCleaner):
    """Remove Edit/Update markers"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'^\s*Edit:?\s*', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*Update:?\s*', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*ETA:?\s*', re.IGNORECASE | re.MULTILINE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


def add_reddit_metadata_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add Reddit metadata cleaners"""
    
    flair_cleaner = RedditFlairCleaner()
    label_cleaner = RedditPostLabelCleaner()
    timestamp_cleaner = MatchTimestampCleaner()
    artifact_cleaner = RedditArtifactCleaner()
    edit_cleaner = EditMarkerCleaner()
    
    cleaners_in_order = [
        flair_cleaner,
        label_cleaner,
        timestamp_cleaner,
        artifact_cleaner,
        edit_cleaner,
    ]
    
    for cleaner in reversed(cleaners_in_order):
        pipeline.cleaners['Reddit'].insert(0, cleaner)

class VerifiedBadgeCleaner(BaseCleaner):
    """Remove verified badges and checkmarks"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'✓', re.UNICODE),
            re.compile(r'☑️', re.UNICODE),
            re.compile(r'\bverified\b', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class FollowerCountCleaner(BaseCleaner):
    """Remove follower/following counts AND surrounding separators"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\|\s*\d+\.?\d*[KMB]?\s*followers?\s*\|?', re.IGNORECASE),
            re.compile(r'\|\s*\d+\.?\d*[KMB]?\s*following\s*\|?', re.IGNORECASE),
            re.compile(r'\b\d+\.?\d*[KMB]?\s*followers?\b', re.IGNORECASE),
            re.compile(r'\b\d+\.?\d*[KMB]?\s*following\b', re.IGNORECASE),
            re.compile(r'\bfollowers?:\s*\d+\.?\d*[KMB]?\b', re.IGNORECASE),
            re.compile(r'\s*\|\s*$'),  # Remove trailing pipes
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text.strip()


class TwitterMetadataCleaner(BaseCleaner):
    """Remove remaining Twitter metadata"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\bjoined\s+\w+\s+\d{4}\s*-?\s*', re.IGNORECASE),
            re.compile(r'\bborn on\s+\w+\s+\d+', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


def add_twitter_metadata_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add Twitter-specific metadata cleaners"""
    
    verified_cleaner = VerifiedBadgeCleaner()
    follower_cleaner = FollowerCountCleaner()
    metadata_cleaner = TwitterMetadataCleaner()
    
    cleaners_in_order = [
        verified_cleaner,
        follower_cleaner,
        metadata_cleaner,
    ]
    
    for cleaner in reversed(cleaners_in_order):
        pipeline.cleaners['Twitter'].insert(0, cleaner)


# Follow me !! 10K followers - me!!

import html as html_module

class HTMLEntityDecoder(BaseCleaner):
    """Decode HTML entities like &amp; &nbsp; &quot;"""
    
    def clean(self, text: str) -> str:
        return html_module.unescape(text)


class HTMLTagCleaner(BaseCleaner):
    """Remove all HTML tags"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<style[^>]*>.*?</style>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<[^>]+>', re.IGNORECASE),
            re.compile(r'&\w+;'),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class HTMLMediaTagCleaner(BaseCleaner):
    """Remove HTML media tags (img, video, audio, iframe)"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'<img[^>]*/?>', re.IGNORECASE),
            re.compile(r'<video[^>]*>.*?</video>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<audio[^>]*>.*?</audio>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<picture[^>]*>.*?</picture>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<figure[^>]*>.*?</figure>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<source[^>]*/?>', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


def add_html_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add HTML cleaners to factual sources"""
    
    entity_decoder = HTMLEntityDecoder()
    tag_cleaner = HTMLTagCleaner()
    media_cleaner = HTMLMediaTagCleaner()
    
    cleaners_in_order = [
        media_cleaner,
        tag_cleaner,
        entity_decoder,
    ]
    
    for cleaner in reversed(cleaners_in_order):
        pipeline.cleaners['Wikipedia'].insert(0, cleaner)
        pipeline.cleaners['NewsAPI'].insert(0, cleaner)
        
class WikipediaCitationCleaner(BaseCleaner):
    """Remove Wikipedia citation markers [1], [2], [citation needed]"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\[\d+\]'),  # Remove [1], [2], etc.
            re.compile(r'\[citation needed\]', re.IGNORECASE),
            re.compile(r'\[clarification needed\]', re.IGNORECASE),
            re.compile(r'\[dubious\]', re.IGNORECASE),
            re.compile(r'\[verification needed\]', re.IGNORECASE),
            re.compile(r'\[original research\?\]', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text

class WikipediaNavigationCleaner(BaseCleaner):
    """Remove Wikipedia navigation text and section markers"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'Main article:\s*[^\n]+', re.IGNORECASE),
            re.compile(r'See also:\s*[^\n]+', re.IGNORECASE),
            re.compile(r'Further information:\s*[^\n]+', re.IGNORECASE),
            re.compile(r'Click to expand', re.IGNORECASE),
            re.compile(r'\(help\)', re.IGNORECASE),
            re.compile(r'\(listen\)', re.IGNORECASE),
            re.compile(r'=+\s*[\w\s]+\s*=+'),  # FIXED: Escaped = symbols
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text
class WikipediaFooterCleaner(BaseCleaner):
    """Remove Wikipedia footer (license, copyright, trademark)"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'Text is available under the.*?non-profit organization\.', re.IGNORECASE | re.DOTALL),
            re.compile(r'licensed under.*?license', re.IGNORECASE),
            re.compile(r'additional terms may apply', re.IGNORECASE),
            re.compile(r'wikipedia.*?is a.*?trademark of.*?(?=\.|$)', re.IGNORECASE),
            re.compile(r',?\s*a non-profit organization', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


def add_wikipedia_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add Wikipedia-specific cleaners"""
    
    citation_cleaner = WikipediaCitationCleaner()
    navigation_cleaner = WikipediaNavigationCleaner()
    footer_cleaner = WikipediaFooterCleaner()
    
    cleaners_in_order = [
        citation_cleaner,
        navigation_cleaner,
        footer_cleaner,
    ]
    
    for cleaner in reversed(cleaners_in_order):
        pipeline.cleaners['Wikipedia'].insert(0, cleaner)

class BylineCleaner(BaseCleaner):
    """Remove author bylines from news articles"""
    
    def _compile_patterns(self):
        self.patterns = [
            # Match "By Name Name" or "By Name Name, Title" followed by period or dash
            re.compile(r'^\s*By\s+[A-Z][\w]+(?:\s+[A-Z][\w]+)*(?:,\s*[\w\s]+)?\.?\s*', re.IGNORECASE),
            re.compile(r'^\s*Author:\s*[\w\s]+\.?\s*', re.IGNORECASE),
            re.compile(r'^\s*Written by\s+[\w\s]+\.?\s*', re.IGNORECASE),
            re.compile(r'^\s*Staff\s+(Writer|Reporter)[\w\s]*\.?\s*', re.IGNORECASE),
            re.compile(r'\s*[-—]\s*[\w\s]+,\s*(CNN|Reuters|AP|BBC|NBC)\s*$', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text

class DatelineCleaner(BaseCleaner):
    """Remove location datelines (NEW YORK —)"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'^[A-Z\s]+\s*[—-]\s*', re.MULTILINE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub('', text)

class UpdateTimestampCleaner(BaseCleaner):
    """Remove update/published timestamps"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'Updated:?\s*[\w\s,]+(?:ago|\d{4})\.?\s*', re.IGNORECASE),  # Added \.?\s*
            re.compile(r'Published:?\s*[\w\s,]+(?:ago|\d{4})\.?\s*', re.IGNORECASE),
            re.compile(r'Last updated:?\s*[\w\s,]+\.?\s*', re.IGNORECASE),
            re.compile(r'\d+\s*(?:hours?|mins?|days?)\s*ago\.?\s*', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text
class NewsTruncationCleaner(BaseCleaner):
    """Clean truncation artifacts"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\.\.\.$'),
            re.compile(r'\[…\]'),
            re.compile(r'…$'),
        ]
    
    def clean(self, text: str) -> str:
        text = self.patterns[0].sub('.', text)
        text = self.patterns[1].sub('', text)
        text = self.patterns[2].sub('.', text)
        return text


class NewsArtifactCleaner(BaseCleaner):
    """Remove news-specific artifacts"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'Read more at.*', re.IGNORECASE),
            re.compile(r'Subscribe to.*', re.IGNORECASE),
            re.compile(r'Related articles?:.*', re.IGNORECASE),
            re.compile(r'Continue reading.*', re.IGNORECASE),
            re.compile(r'Full story at.*', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class OpinionLabelCleaner(BaseCleaner):
    """Remove opinion/editorial/analysis labels"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'^\s*Opinion:\s*', re.IGNORECASE),
            re.compile(r'^\s*Editorial:\s*', re.IGNORECASE),
            re.compile(r'^\s*Analysis:\s*', re.IGNORECASE),
            re.compile(r'^\s*Commentary:\s*', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


def add_news_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add NewsAPI-specific cleaners"""
    
    byline_cleaner = BylineCleaner()
    dateline_cleaner = DatelineCleaner()
    timestamp_cleaner = UpdateTimestampCleaner()
    truncation_cleaner = NewsTruncationCleaner()
    artifact_cleaner = NewsArtifactCleaner()
    opinion_cleaner = OpinionLabelCleaner()
    
    # NewsAPI gets all news cleaners
    cleaners_in_order = [
        byline_cleaner,
        dateline_cleaner,
        timestamp_cleaner,
        truncation_cleaner,
        artifact_cleaner,
        opinion_cleaner,
    ]
    
    for cleaner in reversed(cleaners_in_order):
        pipeline.cleaners['NewsAPI'].insert(0, cleaner)
    
    # ADD: Other sources can have bylines too (quoted news in tweets/reddit)
    pipeline.cleaners['Twitter'].insert(0, byline_cleaner)
    pipeline.cleaners['Reddit'].insert(0, byline_cleaner)
    pipeline.cleaners['Wikipedia'].insert(0, byline_cleaner)

class MediaPlaceholderCleaner(BaseCleaner):
    """Remove image/video/audio placeholders"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\[Image\]', re.IGNORECASE),
            re.compile(r'\[Photo\]', re.IGNORECASE),
            re.compile(r'\[Picture\]', re.IGNORECASE),
            re.compile(r'\[Video\]', re.IGNORECASE),
            re.compile(r'\[Audio\]', re.IGNORECASE),
            re.compile(r'\[Podcast\]', re.IGNORECASE),
            re.compile(r'\[Watch Video\]', re.IGNORECASE),
            re.compile(r'\[Listen\]', re.IGNORECASE),
            re.compile(r'\[Download\]', re.IGNORECASE),
            re.compile(r'\[Attachment\]', re.IGNORECASE),
            re.compile(r'\[File\]', re.IGNORECASE),
            re.compile(r'\(Image\)|\(Photo\)|\(Video\)', re.IGNORECASE),
            re.compile(r'GIF:', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class CaptionMarkerCleaner(BaseCleaner):
    """Remove caption markers but keep caption text"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\b(image caption|photo caption|caption|photo|image|picture):\s*', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text

class CreditLineCleaner(BaseCleaner):
    """Remove photo/image credit lines"""
    
    def _compile_patterns(self):
        self.patterns = [
            # Standard credit formats
            re.compile(r'(photo by|image by|courtesy of|source):\s*[\w\s\.]+?\.', re.IGNORECASE),
            
            # "Credit: Name Date" format
            re.compile(r'\bcredit:\s*[\w\s,]+?(?:\d{1,2}\s+\w+\s+\d{4}[^\w]*)?', re.IGNORECASE),
            
            # Photo agencies (expanded)
            re.compile(r'\b(getty images?|shutterstock|istock|reuters|ap photo|afp|imago)\b', re.IGNORECASE),
            
            # Agency names in parentheses (common format)
            re.compile(r'\([^)]*(?:getty|reuters|ap|afp|imago)[^)]*\)', re.IGNORECASE),
            
            # Camera emoji
            re.compile(r'📸\s*[\w\s]+', re.UNICODE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text
    

class VideoMarkerCleaner(BaseCleaner):
    """Remove video-specific markers"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\bwatch now:?\s*', re.IGNORECASE),
            re.compile(r'\bplay video:?\s*', re.IGNORECASE),
            re.compile(r'\[LIVE\]', re.IGNORECASE),
            re.compile(r'\bstreaming now:?\s*', re.IGNORECASE),
            re.compile(r'\bwatch live:?\s*', re.IGNORECASE),
            re.compile(r'Duration:\s*\d+:\d+', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text
def add_media_placeholder_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add media placeholder cleaners to all sources"""
    
    placeholder_cleaner = MediaPlaceholderCleaner()
    caption_cleaner = CaptionMarkerCleaner()
    credit_cleaner = CreditLineCleaner()
    video_marker_cleaner = VideoMarkerCleaner()
    
    cleaners_in_order = [
        placeholder_cleaner,
        caption_cleaner,
        credit_cleaner,
        video_marker_cleaner,
    ]
    
    # Add to all sources (media placeholders can appear anywhere)
    for source in ['Wikipedia', 'NewsAPI', 'Reddit', 'Twitter']:
        for cleaner in reversed(cleaners_in_order):
            pipeline.cleaners[source].insert(0, cleaner)

class CTAButtonCleaner(BaseCleaner):
    """Remove call-to-action button text"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\b(click|tap|press) here\.?', re.IGNORECASE),
            re.compile(r'\b(read|learn|see|show|view|load|find out) more\.?', re.IGNORECASE),
            re.compile(r'\b(get started|sign up|register|join now)\.?', re.IGNORECASE),
            re.compile(r'\b(download|install|try now|buy now)\.?', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class NavigationCleaner(BaseCleaner):
    """Remove navigation elements"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\b(next|previous) (page|article|post)\.?', re.IGNORECASE),
            re.compile(r'\bback to top\.?', re.IGNORECASE),
            re.compile(r'\bskip to (content|main)\.?', re.IGNORECASE),
            re.compile(r'[«»‹›←→]'),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class StatusLabelCleaner(BaseCleaner):
    """Remove status labels [NEW], [BREAKING], [LIVE], (BETA)"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\[(new|updated|breaking|live|beta|alpha|exclusive)\]', re.IGNORECASE),
            re.compile(r'\((new|updated|beta|alpha|preview)\)', re.IGNORECASE),
            re.compile(r'\b(breaking|live|exclusive):\s*', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class SelectionMarkerCleaner(BaseCleaner):
    """Remove radio buttons and checkbox markers"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'[○●◯⦿☐☑☒✓✗✔✘]\s*'),
            re.compile(r'\[\s*[xX✓]?\s*\]'),
            re.compile(r'\b(option|choice)\s*[A-D]:\s*', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class CookieBannerCleaner(BaseCleaner):
    """Remove cookie consent and privacy banner text"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'(accept|manage) cookies?\.?', re.IGNORECASE),
            re.compile(r'we use cookies.*?\.', re.IGNORECASE),
            re.compile(r'by continuing.*?cookies.*?\.', re.IGNORECASE),
            re.compile(r'(privacy policy|terms of service|cookie policy)\.?', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text

class PaywallTextCleaner(BaseCleaner):
    """Remove paywall text"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'sign in to read more', re.IGNORECASE),  # Specific phrase first
            re.compile(r'sign in to\b', re.IGNORECASE),  # Then general "sign in to"
            re.compile(r'this content is for subscribers only', re.IGNORECASE),
            re.compile(r'member[- ]exclusive', re.IGNORECASE),
            re.compile(r'\d+\s*free articles? remaining', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text

def add_ui_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add UI element cleaners"""
    
    cta_cleaner = CTAButtonCleaner()
    nav_cleaner = NavigationCleaner()
    status_cleaner = StatusLabelCleaner()
    selection_cleaner = SelectionMarkerCleaner()
    cookie_cleaner = CookieBannerCleaner()
    paywall_cleaner = PaywallTextCleaner()
    
    cleaners_in_order = [
        cta_cleaner,
        nav_cleaner,
        status_cleaner,
        selection_cleaner,
        cookie_cleaner,
        paywall_cleaner,
    ]
    
    # Add to NewsAPI, Reddit, and Twitter
    for source in ['NewsAPI', 'Reddit', 'Twitter']:
        for cleaner in reversed(cleaners_in_order):
            pipeline.cleaners[source].insert(0, cleaner)

def add_markdown_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add markdown cleaners"""
    
    table_cleaner = RedditTableCleaner()  # Has this
    double_hash_cleaner = DoubleHashCleaner()  # Has this
    header_cleaner = MarkdownHeaderCleaner()
    bold_italic_cleaner = BoldItalicCleaner()
    quote_cleaner = QuoteMarkerCleaner()
    bullet_cleaner = BulletPointCleaner()
    code_cleaner = CodeBlockCleaner()
    
    cleaners_in_order = [
        code_cleaner,
        table_cleaner,  # NEW
        double_hash_cleaner,  # NEW
        header_cleaner,
        bold_italic_cleaner,
        quote_cleaner,
        bullet_cleaner,
    ]
    
    # Reddit gets all
    for cleaner in reversed(cleaners_in_order):
        pipeline.cleaners['Reddit'].insert(0, cleaner)
    
    # Others get essential ones
    for source in ['NewsAPI', 'Twitter', 'Wikipedia']:
        pipeline.cleaners[source].insert(0, bullet_cleaner)
        pipeline.cleaners[source].insert(0, quote_cleaner)
        pipeline.cleaners[source].insert(0, bold_italic_cleaner)
        pipeline.cleaners[source].insert(0, header_cleaner)
        pipeline.cleaners[source].insert(0, double_hash_cleaner)
        pipeline.cleaners[source].insert(0, table_cleaner)  # NEW

class AdvertisementCleaner(BaseCleaner):
    """Remove advertisement labels and detect full-ad content"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\badvertisement\b', re.IGNORECASE),
            re.compile(r'\[(ad|sponsored|promoted)\]', re.IGNORECASE),
            re.compile(r'\((ad|sponsored)\)', re.IGNORECASE),
            re.compile(r'\bsponsored content\b', re.IGNORECASE),
            re.compile(r'\bpaid promotion\b', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class BettingPromoCleaner(BaseCleaner):
    """Remove betting promotional text - only if clearly gambling related"""
    
    def _compile_patterns(self):
        # Only match if it has betting company OR "bonus bets/code" specifically
        self.full_promo = re.compile(
            r'.*?\b(draftkings|bet365|fanduel|betmgm|caesars)\b.*|'
            r'.*?\b(bonus bets|bonus code|promo code)\b.*|'
            r'.*?bet\s+\$\d+.*',
            re.IGNORECASE
        )
    
    def clean(self, text: str) -> str:
        if self.full_promo.match(text):
            return ""
        return text
    

class TradingAlertCleaner(BaseCleaner):
    """Remove trading/betting alert spam"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'Position:\s*No\s*@\s*\d+%', re.IGNORECASE),
            re.compile(r'Track smart money live', re.IGNORECASE),
            re.compile(r'SMART MONEY ALERT', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text
    

class DiscountCleaner(BaseCleaner):
    """Remove discount text"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'-?\d+%\s*off', re.IGNORECASE),
            re.compile(r'\d+%\s*discount', re.IGNORECASE),
            re.compile(r'\b(special offer|flash sale|limited time|clearance)', re.IGNORECASE),
            re.compile(r'\bup to\s+\d+%\s*off', re.IGNORECASE),
            re.compile(r'\bbuy \d+ get \d+ free', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class PriceCleaner(BaseCleaner):
    """Remove prices"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\$\d+(?:\.\d{2})?'),
            re.compile(r'[€£]\d+(?:\.\d{2})?'),
            re.compile(r'\b\d+(?:\.\d{2})?\s*(dollars?|usd|eur|gbp)', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class ShoppingCTACleaner(BaseCleaner):
    """Remove shopping CTAs"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\b(shop|buy|order) (now|today)', re.IGNORECASE),
            re.compile(r'\bavailable at\s+\w+', re.IGNORECASE),
            re.compile(r'\bfree shipping', re.IGNORECASE),
            re.compile(r'\bwhile supplies last', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class SubscriptionCleaner(BaseCleaner):
    """Remove subscription offers"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\bsupport\s+\$\d+/?(monthly|month|year)?', re.IGNORECASE),
            re.compile(r'\bunlock\s+[\w-]+\s+benefits?', re.IGNORECASE),
            re.compile(r'\brecommended\b', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


def add_promotional_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add promotional cleaners - betting runs FIRST to catch full promos"""
    
    betting_cleaner = BettingPromoCleaner()  # FIRST - catches full betting ads
    ad_cleaner = AdvertisementCleaner()
    trading_alert_cleaner = TradingAlertCleaner()
    discount_cleaner = DiscountCleaner()
    price_cleaner = PriceCleaner()
    shopping_cleaner = ShoppingCTACleaner()
    subscription_cleaner = SubscriptionCleaner()
    
    cleaners_in_order = [
        betting_cleaner,  # Run first to remove entire betting ads
        ad_cleaner,
        trading_alert_cleaner, 
        subscription_cleaner,
        discount_cleaner,
        shopping_cleaner,
        price_cleaner,
    ]
    
    for source in ['Wikipedia', 'NewsAPI', 'Reddit', 'Twitter']:
        for cleaner in reversed(cleaners_in_order):
            pipeline.cleaners[source].insert(0, cleaner)

class CopyrightCleaner(BaseCleaner):
    """Remove copyright notices"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'©\s*\d{4}.*?(?=\.|$)', re.IGNORECASE),
            re.compile(r'\bcopyright\s+©?\s*\d{4}.*?(?=\.|$)', re.IGNORECASE),
            re.compile(r'\ball rights reserved\.?', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class LicenseCleaner(BaseCleaner):
    """Remove license text"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'text is available under the.*?license\.', re.IGNORECASE | re.DOTALL),
            re.compile(r'licensed under.*?(?=\.|$)', re.IGNORECASE),
            re.compile(r'creative commons.*?license', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class TermsPrivacyCleaner(BaseCleaner):
    """Remove privacy policy and terms of service links"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\b(privacy policy|terms of (use|service)|cookie (policy|statement))\.?', re.IGNORECASE),
            re.compile(r'additional terms may apply\.?', re.IGNORECASE),
            re.compile(r'by using this site,.*?(?=\.|$)', re.IGNORECASE),
            re.compile(r'you agree to the.*?(?=\.|$)', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class TrademarkCleaner(BaseCleaner):
    """Remove trademark text"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\w+®?\s*is a (registered )?trademark of.*?(?=\.|$)', re.IGNORECASE),  # Added \w+
            re.compile(r',?\s*a non-profit (organization|organisation)\.?', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text
        
class FooterNavigationCleaner(BaseCleaner):
    """Remove footer navigation clusters"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\bmobile view\.?', re.IGNORECASE),
            re.compile(r'\bcookie statement\.?', re.IGNORECASE),
            re.compile(r'\blegal & safety contacts\.?', re.IGNORECASE),
            re.compile(r'\bpowered by \w+\.?', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


def add_footer_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add footer and legal text cleaners"""
    
    copyright_cleaner = CopyrightCleaner()
    license_cleaner = LicenseCleaner()
    terms_cleaner = TermsPrivacyCleaner()
    trademark_cleaner = TrademarkCleaner()
    footer_nav_cleaner = FooterNavigationCleaner()
    
    cleaners_in_order = [
        copyright_cleaner,
        license_cleaner,
        terms_cleaner,
        trademark_cleaner,
        footer_nav_cleaner,
    ]
    
    # Add to all sources (footers can appear anywhere)
    for source in ['Wikipedia', 'NewsAPI', 'Reddit', 'Twitter']:
        for cleaner in reversed(cleaners_in_order):
            pipeline.cleaners[source].insert(0, cleaner)

class TimestampCleaner(BaseCleaner):
    """Remove timestamp patterns"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\b\d+\s*(?:hours?|hrs?|mins?|minutes?|days?|weeks?|months?|years?)\s+ago\.?\s*', re.IGNORECASE),
            re.compile(r'\bjust now\.?\s*', re.IGNORECASE),
            re.compile(r'\byesterday\.?\s*', re.IGNORECASE),
            re.compile(r'\bposted:?\s*', re.IGNORECASE),
            re.compile(r'\bpublished:?\s*', re.IGNORECASE),
            re.compile(r'\bupdated:?\s*', re.IGNORECASE),
            re.compile(r'\blast updated:?\s*', re.IGNORECASE),
            re.compile(r'\bat\s+\d{1,2}:\d{2}\s*(?:AM|PM|EST|PST|UTC)?\.?\s*', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class LocationTagCleaner(BaseCleaner):
    """Remove ONLY formatted location tags, keep location in natural text"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'📍\s*', re.UNICODE),  # Just remove pin, keep location name
            re.compile(r'\blocation:\s*', re.IGNORECASE),  # Just remove "Location:", keep name
            re.compile(r'\bfrom:\s*(?=[\w\s,]+(?:\.|$))', re.IGNORECASE),  # Remove "From:" only
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text

class ViewCountCleaner(BaseCleaner):
    """Remove view counts"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\bviewed\s+\d+\.?\d*[KMB]?\s*times?\.?\s*', re.IGNORECASE),
            re.compile(r'^\s*\d+\.?\d*[KMB]?\s*views?\s*$', re.IGNORECASE | re.MULTILINE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class CoordinateCleaner(BaseCleaner):
    """Remove geographic coordinates"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'coordinates?:\s*\d+\.?\d*°\s*[NS],?\s*\d+\.?\d*°\s*[EW]\.?\s*', re.IGNORECASE)
    
    def clean(self, text: str) -> str:
        return self.pattern.sub('', text)


def add_metadata_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add metadata cleaners to all sources"""
    
    timestamp_cleaner = TimestampCleaner()
    location_cleaner = LocationTagCleaner()
    view_count_cleaner = ViewCountCleaner()
    coordinate_cleaner = CoordinateCleaner()
    
    cleaners_in_order = [
        timestamp_cleaner,
        location_cleaner,
        view_count_cleaner,
        coordinate_cleaner,
    ]
    
    for source in ['Wikipedia', 'NewsAPI', 'Reddit', 'Twitter']:
        for cleaner in reversed(cleaners_in_order):
            pipeline.cleaners[source].insert(0, cleaner)


class ExcessivePunctuationCleaner(BaseCleaner):
    """Normalize excessive punctuation (!!!!! → !)"""
    
    def _compile_patterns(self):
        self.patterns = [
            (re.compile(r'!{2,}'), '!'),
            (re.compile(r'\?{2,}'), '?'),
            (re.compile(r'\.{4,}'), '...'),
            (re.compile(r'!+\?+|[\?!]{3,}'), '?!'),
        ]
    
    def clean(self, text: str) -> str:
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        return text


class InvisibleCharCleaner(BaseCleaner):
    """Remove invisible Unicode characters"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\u200b'),  # Zero-width space
            re.compile(r'\u200c'),  # Zero-width non-joiner
            re.compile(r'\u200d'),  # Zero-width joiner
            re.compile(r'\ufeff'),  # Zero-width no-break space
            re.compile(r'\u202a|\u202b|\u202c|\u202d|\u202e'),  # Unicode direction markers
            re.compile(r'\xa0'),    # Non-breaking space
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub(' ', text)
        return text


class ASCIIArtCleaner(BaseCleaner):
    """Remove ASCII art and repeated characters"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'(.)\1{10,}'),  # Same character repeated 10+ times
            re.compile(r'[-=_]{5,}'),   # Separator lines
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class SpacingNormalizer(BaseCleaner):
    """Advanced spacing fixes"""
    
    def _compile_patterns(self):
        self.patterns = [
            (re.compile(r'\s+([.,!?;:])'), r'\1'),       # Fix spacing before punctuation
            (re.compile(r'([.,!?;:])\s*([.,!?;:])'), r'\1'),  # Remove double punctuation
            (re.compile(r'\(\s+'), '('),                 # Fix "( text" → "(text"
            (re.compile(r'\s+\)'), ')'),                 # Fix "text )" → "text)"
            (re.compile(r'\[\s+'), '['),                 # Fix "[ text" → "[text"
            (re.compile(r'\s+\]'), ']'),                 # Fix "text ]" → "text]"
        ]
    
    def clean(self, text: str) -> str:
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        return text


class EmptyBracketCleaner(BaseCleaner):
    """Remove empty brackets AND common placeholder text in brackets"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\(\s*\)'),
            re.compile(r'\[\s*\]'),
            re.compile(r'\{\s*\}'),
            # ADD: Remove common placeholder text
            re.compile(r'\(empty\)', re.IGNORECASE),
            re.compile(r'\(none\)', re.IGNORECASE),
            re.compile(r'\(null\)', re.IGNORECASE),
            re.compile(r'\(n/a\)', re.IGNORECASE),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text

def add_normalization_cleaners_to_pipeline(pipeline: TextCleaningPipeline):
    """Add text normalization cleaners to all sources"""
    
    punctuation_cleaner = ExcessivePunctuationCleaner()
    invisible_cleaner = InvisibleCharCleaner()
    ascii_cleaner = ASCIIArtCleaner()
    spacing_cleaner = SpacingNormalizer()
    bracket_cleaner = EmptyBracketCleaner()
    date_normalizer = DateNormalizer() 

    
    cleaners_in_order = [
        invisible_cleaner,
        ascii_cleaner,
        punctuation_cleaner,
        bracket_cleaner,
         date_normalizer, 
        spacing_cleaner,
    ]
    
    for source in ['Wikipedia', 'NewsAPI', 'Reddit', 'Twitter']:
        for cleaner in reversed(cleaners_in_order):
            pipeline.cleaners[source].insert(0, cleaner)

class EmptyParenthesesCleaner(BaseCleaner):
    """Remove empty parentheses/brackets left after cleaning"""
    
    def _compile_patterns(self):
        self.patterns = [
            re.compile(r'\(\s*\)'),
            re.compile(r'\[\s*\]'),
            re.compile(r'\{\s*\}'),
            re.compile(r'<\s*>'),
        ]
    
    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub('', text)
        return text


class OrphanedPunctuationCleaner(BaseCleaner):
    """Clean orphaned punctuation and spacing issues"""
    
    def _compile_patterns(self):
        self.patterns = [
            (re.compile(r'\s+,'), ','),           # Fix " ," → ","
            (re.compile(r'\s+\.'), '.'),          # Fix " ." → "."
            (re.compile(r'\s+;'), ';'),           # Fix " ;" → ";"
            (re.compile(r'\s+:'), ':'),           # Fix " :" → ":"
            (re.compile(r'^\s*[,;:\.]\s*'), ''),  # Remove punctuation at start
            (re.compile(r'\s+$'), ''),            # Remove trailing whitespace
        ]
    
    def clean(self, text: str) -> str:
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        return text.strip()


class DuplicateWordCleaner(BaseCleaner):
    """Remove accidentally duplicated words"""
    
    def _compile_patterns(self):
        self.pattern = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
    
    def clean(self, text: str) -> str:
        # Only remove if same word appears twice in a row
        return self.pattern.sub(r'\1', text)


class FinalWhitespaceCleaner(BaseCleaner):
    """Final aggressive whitespace cleanup"""
    
    def _compile_patterns(self):
        self.patterns = [
            (re.compile(r'\s+'), ' '),                    # Collapse all whitespace
            (re.compile(r'^\s+|\s+$'), ''),              # Trim start/end
            (re.compile(r'\s+([.,!?;:])'), r'\1'),       # Fix spacing before punctuation
            (re.compile(r'([.,!?;:])\s*([.,!?;:])'), r'\1'),  # Remove double punctuation
        ]
    
    def clean(self, text: str) -> str:
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        return text.strip()


class MinimumLengthValidator(BaseCleaner):
    """Remove text that's too short to be meaningful"""
    
    def clean(self, text: str) -> str:
        # If after all cleaning, text is < 5 chars, mark as empty
        if len(text.strip()) < 5:
            return ""
        return text

class FinalHTMLEntityDecoder(BaseCleaner):
    """Final pass to catch any remaining HTML entities"""
    
    def clean(self, text: str) -> str:
        import html
        # Decode twice to catch double-encoded entities
        text = html.unescape(text)
        text = html.unescape(text)
        return text
    
class DateNormalizer(BaseCleaner):
    """Convert dates to natural language format"""
    
    def _compile_patterns(self):
        self.date_pattern = re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b')
        self.month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
    
    def clean(self, text: str) -> str:
        def replace_date(match):
            month, day, year = match.groups()
            try:
                month_int = int(month)
                day_int = int(day)
                year_int = int(year)
                
                # Skip if invalid (catches ratings like 2/10)
                if month_int > 12 or day_int > 31 or month_int == 0 or day_int == 0:
                    return match.group(0)
                
                # Handle 2-digit years
                if year_int < 100:
                    year_int = 2000 + year_int if year_int < 50 else 1900 + year_int
                
                # Validate it's a real date
                from datetime import datetime
                datetime(year_int, month_int, day_int)
                
                month_name = self.month_names[month_int]
                return f"{month_name} {day_int}, {year_int}"
                    
            except (ValueError, KeyError):
                return match.group(0)
        
        return self.date_pattern.sub(replace_date, text)

def add_final_polish_to_pipeline(pipeline: TextCleaningPipeline):
    """Add final polish cleaners - run at the very end"""
    
    empty_paren_cleaner = EmptyParenthesesCleaner()
    orphaned_punct_cleaner = OrphanedPunctuationCleaner()
    duplicate_word_cleaner = DuplicateWordCleaner()
    final_whitespace_cleaner = FinalWhitespaceCleaner()
    final_html_decoder = FinalHTMLEntityDecoder()  
    min_length_validator = MinimumLengthValidator()
    
    cleaners_in_order = [
        empty_paren_cleaner,
        orphaned_punct_cleaner,
        duplicate_word_cleaner,
        final_html_decoder, 
        final_whitespace_cleaner,
        min_length_validator,
    ]
    
    # FIXED: Increment insert position for each cleaner
    for source in ['Wikipedia', 'NewsAPI', 'Reddit', 'Twitter']:
        insert_position = len(pipeline.cleaners[source]) - 1
        for i, cleaner in enumerate(cleaners_in_order):
            pipeline.cleaners[source].insert(insert_position + i, cleaner)
            # insert_position + 0 = position 3 (first cleaner)
            # insert_position + 1 = position 4 (second cleaner)
            # insert_position + 2 = position 5 (third cleaner)
            # etc.

def clean_topic_column(df, pipeline):
    """Clean topic column with simplified approach"""
    
    # Topics are usually short and clean, just remove obvious noise
    def clean_topic(topic, source):
        if pd.isna(topic):
            return topic
        
        # Just run core cleaners
        text = topic
        
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        
        # Clean hashtags
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Clean mentions  
        text = re.sub(r'@(\w+)', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    df['topic_cleaned'] = df.apply(
        lambda row: clean_topic(row['topic'], row['source']),
        axis=1
    )
    
    return df

def setup_cleaning_pipeline():
    pipeline = TextCleaningPipeline()
    
    whitespace = WhitespaceCleaner()
    encoding_fixer = UniversalEncodingFixer()
    
    for source in pipeline.cleaners.keys():
        pipeline.cleaners[source].append(whitespace)

    for source in pipeline.cleaners.keys():
        pipeline.cleaners[source].insert(0, encoding_fixer)
    
    # REVERSE THIS ORDER - add URL cleaners LAST so they run FIRST
    
    add_final_polish_to_pipeline(pipeline)
    add_normalization_cleaners_to_pipeline(pipeline)
    add_metadata_cleaners_to_pipeline(pipeline)
    add_footer_cleaners_to_pipeline(pipeline)
    add_promotional_cleaners_to_pipeline(pipeline)
    add_ui_cleaners_to_pipeline(pipeline)
    add_media_placeholder_cleaners_to_pipeline(pipeline)
    add_news_cleaners_to_pipeline(pipeline)
    add_wikipedia_cleaners_to_pipeline(pipeline)
    add_html_cleaners_to_pipeline(pipeline)
    add_twitter_metadata_cleaners_to_pipeline(pipeline)
    add_reddit_metadata_cleaners_to_pipeline(pipeline)
    add_markdown_cleaners_to_pipeline(pipeline)
    add_emoji_cleaners_to_pipeline(pipeline)
    add_hashtag_cleaners_to_pipeline(pipeline)    
    add_interaction_cleaners_to_pipeline(pipeline)
    add_url_cleaners_to_pipeline(pipeline)         
    return pipeline
