# Glenda A

# useful libraries or modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ChatbotAnalyzer:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def prepare_data(self, conversations_df):
        """
        Prepare conversation data for analysis
        
        Parameters:
        conversations_df: DataFrame with columns 'user_input', 'bot_response', 'intent'
        """
        # Tokenize user inputs
        self.tokenizer.fit_on_texts(conversations_df['user_input'])
        X = self.tokenizer.texts_to_sequences(conversations_df['user_input'])
        X = pad_sequences(X, maxlen=50)
        
        # Encode intents
        y = self.label_encoder.fit_transform(conversations_df['intent'])
        
        return X, y
    
    def build_model(self, vocab_size, num_classes):
        """
        Build LSTM model for intent prediction
        """
        self.model = Sequential([
            Embedding(vocab_size, 100, input_length=50),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def analyze_interactions(self, conversations_df):
        """
        Analyze chatbot interactions
        """
        analysis = {
            'total_conversations': len(conversations_df),
            'unique_intents': conversations_df['intent'].nunique(),
            'avg_user_msg_length': conversations_df['user_input'].str.len().mean(),
            'intent_distribution': conversations_df['intent'].value_counts()
        }
        
        # Response time analysis (assuming timestamp column exists)
        if 'timestamp' in conversations_df.columns:
            conversations_df['response_time'] = conversations_df.groupby('conversation_id')['timestamp'].diff()
            analysis['avg_response_time'] = conversations_df['response_time'].mean()
        
        return analysis
    
    def evaluate_performance(self, conversations_df):
        """
        Evaluate chatbot performance metrics
        """
        performance = {
            'success_rate': len(conversations_df[conversations_df['success'] == True]) / len(conversations_df),
            'intent_accuracy': len(conversations_df[conversations_df['predicted_intent'] == conversations_df['actual_intent']]) / len(conversations_df),
            'avg_user_satisfaction': conversations_df['user_satisfaction'].mean() if 'user_satisfaction' in conversations_df.columns else None
        }
        
        return performance

def create_sample_data():
    """
    Create sample chatbot conversation data
    """
    data = {
        'user_input': [
            'what are your business hours',
            'how do I reset my password',
            'I want to cancel my order',
            'where is my shipment',
            'can I talk to a human'
        ],
        'bot_response': [
            'We are open Monday-Friday 9AM-5PM',
            'You can reset your password at the login page',
            'I can help you cancel your order',
            'Let me check your shipment status',
            'Transferring you to a human agent'
        ],
        'intent': [
            'hours_inquiry',
            'technical_support',
            'order_cancellation',
            'shipping_status',
            'human_handoff'
        ]
    }
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    # Create sample data
    conversations_df = create_sample_data()
    
    # Initialize analyzer
    analyzer = ChatbotAnalyzer()
    
    # Prepare data for model
    X, y = analyzer.prepare_data(conversations_df)
    
    # Build and train model
    vocab_size = len(analyzer.tokenizer.word_index) + 1
    num_classes = len(np.unique(y))
    model = analyzer.build_model(vocab_size, num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Analyze interactions
    analysis_results = analyzer.analyze_interactions(conversations_df)
    
    # Evaluate performance
    performance_metrics = analyzer.evaluate_performance(conversations_df)
