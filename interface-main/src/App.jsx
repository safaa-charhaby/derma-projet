import { useState } from 'react';
import {
  AlertCircle, Check, X, Loader2, ChevronRight,
  Info, Award, Shield, Star, ShoppingBag
} from 'lucide-react';
import './App.css';

export default function App() {
  // === Product types by face area ===
  const productTypesByArea = {
    Face: ["Overnight Mask","Makeup Remover", ,"Facial Treatment","Serum", "Sunscreen", "Face Cleanser","The Face Shop","Face Makeup","Toner","Sheet Mask"],
    Eyes: ["Eye Cream","Eye Moisturizer","Eye Makeup"],
    Lips: ["Lip Moisturizer", "Lip Makeup","Lip Mask"],
    Hair:["Other Haircare","Conditioner","Shampoo"],
    Body:["Bath & Body","Fragrance"],
    Hand:["Nail Care","Hand Care"],
    General:["General Moisturizer","Oil","Essence","Emulsion","Exfoliator"],
    Cheek:["Cheek Makeup"]
  };

  const faceAreas = Object.keys(productTypesByArea);

  // === State ===
  const [selectedFaceArea, setSelectedFaceArea] = useState(null);
  const [selectedProductType, setSelectedProductType] = useState(null);
  const [selectedConcerns, setSelectedConcerns] = useState(
    Object.fromEntries([
      'Acne Fighting','Acne Trigger','Anti-Aging','Brightening','Dark Spots','Drying',
      'Eczema','Good For Oily Skin','Hydrating','Irritating','Redness Reducing',
      'Reduces Irritation','Reduces Large Pores','Rosacea','Scar Healing'
    ].map(c => [c,false]))
  );
  const [recommendations, setRecommendations] = useState(null);
  const [recommendedProducts, setRecommendedProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [productsLoading, setProductsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [productsError, setProductsError] = useState(null);
  const [activeStep, setActiveStep] = useState(1);

  // === Derived values ===
  const selectedCount = Object.values(selectedConcerns).filter(Boolean).length;
  const productTypes = selectedFaceArea ? productTypesByArea[selectedFaceArea] : [];
  const groupedIngredients = recommendations ? {
    recommended: Object.entries(recommendations).filter(([_,v])=>v==='Yes').map(([k])=>k),
    notRecommended: Object.entries(recommendations).filter(([_,v])=>v==='No').map(([k])=>k)
  } : null;

  // === Handlers ===
  const handleFaceAreaSelect = (area) => {
    setSelectedFaceArea(area);
    setSelectedProductType(null);
  };

  const handleProductTypeSelect = (type) => {
    setSelectedProductType(type);
    setActiveStep(2);
  };

  const handleConcernToggle = (concern) => {
    setSelectedConcerns(prev => ({ ...prev, [concern]: !prev[concern] }));
  };

  const handleSubmit = async () => {
    setLoading(true); setError(null);
    try {
      const features = Object.values(selectedConcerns).map(v => v ? 1 : 0);
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ features, product_type: selectedProductType })
      });
      if (!res.ok) throw new Error('Failed to get recommendations');
      const data = await res.json();
      setRecommendations(data.ingredients);
      setActiveStep(3);
    } catch(err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const getProductRecommendations = async () => {
    if (!recommendations) return;
    setProductsLoading(true); setProductsError(null);
    try {
      const res = await fetch('http://localhost:5000/filter-products', {
        method: 'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ ingredients: recommendations, product_type: selectedProductType })
      });
      if(!res.ok) throw new Error('Failed to get product recommendations');
      const data = await res.json();
      setRecommendedProducts(data.products || []);
      setActiveStep(4);
    } catch(err) { setProductsError(err.message); }
    finally { setProductsLoading(false); }
  };

  const resetForm = () => {
    setSelectedFaceArea(null);
    setSelectedProductType(null);
    setSelectedConcerns(Object.fromEntries(Object.keys(selectedConcerns).map(c => [c,false])));
    setRecommendations(null);
    setRecommendedProducts([]);
    setActiveStep(1);
    setError(null);
    setProductsError(null);
  };

  // === Helpers ===
   const getProductIcon = (type, isSelected = false) => {
    const color = isSelected ? 'text-white' : 'text-rose-600';
    const icons = {
      "Overnight Mask": <ShoppingBag className={`w-6 h-6 ${color}`} />,
      "Makeup Remover": <Shield className={`w-6 h-6 ${color}`} />,
      "Facial Treatment": <Award className={`w-6 h-6 ${color}`} />,
      "Serum": <Award className={`w-6 h-6 ${color}`} />,
      "Sunscreen": <Star className={`w-6 h-6 ${color}`} />,
      "Face Cleanser": <Info className={`w-6 h-6 ${color}`} />,
      "The Face Shop": <ShoppingBag className={`w-6 h-6 ${color}`} />,
      "Face Makeup": <Star className={`w-6 h-6 ${color}`} />,
      "Toner": <Shield className={`w-6 h-6 ${color}`} />,
      "Sheet Mask": <ShoppingBag className={`w-6 h-6 ${color}`} />,
      "Eye Cream": <Info className={`w-6 h-6 ${color}`} />,
      "Eye Moisturizer": <Star className={`w-6 h-6 ${color}`} />,
      "Eye Makeup": <ShoppingBag className={`w-6 h-6 ${color}`} />,
      "Lip Moisturizer": <ShoppingBag className={`w-6 h-6 ${color}`} />,
      "Lip Makeup": <Shield className={`w-6 h-6 ${color}`} />,
      "Lip Mask": <Star className={`w-6 h-6 ${color}`} />,
      "Other Haircare": <Shield className={`w-6 h-6 ${color}`} />,
      "Conditioner": <Award className={`w-6 h-6 ${color}`} />,
      "Shampoo": <Star className={`w-6 h-6 ${color}`} />,
      "Bath & Body": <Shield className={`w-6 h-6 ${color}`} />,
      "Fragrance": <Star className={`w-6 h-6 ${color}`} />,
      "Hand Care": <ShoppingBag className={`w-6 h-6 ${color}`} />,
      "Nail Care": <Star className={`w-6 h-6 ${color}`} />,
      "General Moisturizer": <Star className={`w-6 h-6 ${color}`} />,
      "Oil": <Award className={`w-6 h-6 ${color}`} />,
      "Essence": <Info className={`w-6 h-6 ${color}`} />,
      "Emulsion": <Shield className={`w-6 h-6 ${color}`} />,
      "Exfoliator": <ShoppingBag className={`w-6 h-6 ${color}`} />,
      "Cheek Makeup": <Shield className={`w-6 h-6 ${color}`} />
    };return icons[type] || <ShoppingBag className={`w-6 h-6 ${color}`} />;
  };

  const formatIngredientName = (name) => name.replace(/_/g,' ');
  const getIngredientDescription = (name) => 'This ingredient is suitable for your skin concerns.';
  const getProductIngredients = (product, recommended) => product.ingredients?.filter(i=>recommended.includes(i)) || [];

  return (
    <div className="min-h-screen bg-gradient-to-b from-rose-50 to-white">
      {/* Header */}
      <div className="bg-gradient-to-r from-rose-700 to-pink-600 text-white py-8 md:py-16 shadow-lg">
        <div className="max-w-6xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center">
          <div className="mb-6 md:mb-0 text-center md:text-left">
            <h1 className="text-4xl md:text-5xl font-bold mb-3 font-serif tracking-tight">GlowGuide AI</h1>
            <p className="text-rose-100 text-lg md:text-xl max-w-xl">Discover your perfect skincare routine</p>
          </div>
          <div className="hidden md:block w-20 h-20 bg-white/20 rounded-full flex items-center justify-center backdrop-blur-sm animate-pulse">
            <Star className="w-10 h-10 text-white" />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-8 md:py-12">
        {/* Progress Steps */}
        <div className="mb-12 flex items-center justify-center gap-2 md:gap-6">
          {[1,2,3,4].map(step => (
            <div key={step} className="flex flex-col items-center">
              <div className={`relative w-14 h-14 rounded-full flex items-center justify-center transition-all duration-300 ${activeStep >= step ? 'bg-gradient-to-br from-rose-600 to-pink-500 text-gray shadow-lg' : 'bg-gray-100 text-gray-400'}`}>
                <span className="font-semibold text-lg">{step}</span>
                {activeStep > step && <Check className="absolute -right-1 -top-1 w-5 h-5 bg-green-500 text-gray rounded-full p-1" />}
              </div>
              <span className={`mt-2 text-xs font-medium ${activeStep === step ? 'text-rose-700 font-semibold' : 'text-gray-500'}`}>
                {['Face/Product', 'Concerns', 'Ingredients', 'Products'][step-1]}
              </span>
            </div>
          ))}
        </div>

        {/* Step 1: Face Area & Product Type */}
        {activeStep === 1 && (
          <div className="bg-white rounded-3xl shadow-lg overflow-hidden border border-rose-100">
            <div className="bg-gradient-to-r from-rose-50 to-pink-50 px-8 py-6 border-b border-rose-100">
              <h2 className="text-2xl font-bold text-gray-800 font-serif">
                {selectedFaceArea ? "Select Product Category" : "Select Face Area"}
              </h2>
              <p className="text-rose-600 text-sm mt-1">
                {selectedFaceArea ? `Choose the type of product for your ${selectedFaceArea.toLowerCase()}` : "Choose which part of your face you want to treat"}
              </p>
            </div>
            <div className="p-6 md:p-8">
              {!selectedFaceArea ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                  {faceAreas.map(area => (
                    <button key={area} onClick={() => handleFaceAreaSelect(area)} className="p-4 rounded-xl flex flex-col items-center transition-all duration-200 border-2 bg-white border-gray-200 hover:border-pink-900 hover:bg-pink-50 hover:text-pink-900">
                      <div className="w-12 h-12 rounded-full bg-rose-100 flex items-center justify-center mb-3">🌸</div>
                      <span className="font-medium text-gray-700">{area}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                    {productTypes.map(type => (
                      <button key={type} onClick={() => handleProductTypeSelect(type)}
                        className={`p-4 rounded-xl flex flex-col items-center transition-all duration-200 border-2 ${selectedProductType === type ? "bg-gradient-to-br from-pink-900 to-pink-900 text-white shadow-lg transform scale-105 border-rose-400" : "bg-white border-gray-200 hover:border-pink-900 hover:bg-pink-50 hover:text-pink-900"}`}>
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-3 ${selectedProductType === type ? "bg-white/30" : "bg-rose-100"}`}>
                          {getProductIcon(type, selectedProductType === type)}
                        </div>
                        <span className={`font-medium ${selectedProductType === type ? "text-white" : "text-gray-700"}`}>{type}</span>
                      </button>
                    ))}
                  </div>
                  <div className="mt-6">
                    <button onClick={() => setSelectedFaceArea(null)} className="text-sm text-pink-900 hover:underline">← Change Face Area</button>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Step 2: Concerns */}
        {activeStep === 2 && (
          <div className="bg-white rounded-3xl shadow-lg overflow-hidden border border-rose-100">
            <div className="bg-gradient-to-r from-rose-50 to-pink-50 px-8 py-6 border-b border-rose-100 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-gray-800 font-serif">Your Skin Concerns</h2>
                <p className="text-rose-600 text-sm mt-1">Select all that apply</p>
              </div>
              <div className="bg-gradient-to-r from-yellow-600 to-yellow-600 text-white px-4 py-2 rounded-full text-sm font-semibold flex items-center shadow-lg">{selectedCount} Selected</div>
            </div>
            <div className="p-6 md:p-8">
              <div className="flex items-start mb-6 bg-gradient-to-r from-pink-100 to-pink-100 p-4 rounded-xl border border-gray-200">
                <Info className="text-rose-600 w-5 h-5 mr-3 mt-0.5 flex-shrink-0" />
                <p className="text-rose-800 text-sm">Our AI will analyze your combination to recommend the best {selectedProductType?.toLowerCase()} for your needs.</p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {Object.keys(selectedConcerns).map(concern => (
                  <button key={concern} onClick={() => handleConcernToggle(concern)}
                    className={`p-3 rounded-lg flex items-center gap-3 text-left transition-all duration-150 border ${selectedConcerns[concern] ? 'bg-gradient-to-r from-pink-900 to-pink-900 text-white shadow-md border-pink-900 transform scale-[1.02]' : 'bg-white border-gray-200 hover:border-pink-900 hover:bg-pink-50'}`}>
                    <div className={`w-5 h-5 flex-shrink-0 rounded-md flex items-center justify-center transition-all ${selectedConcerns[concern] ? 'bg-white text-pink-600' : 'border-2 border-gray-300'}`}>
                      {selectedConcerns[concern] && <Check className="w-3 h-3" />}
                    </div>
                    <span className={`font-medium ${selectedConcerns[concern] ? 'text-white' : 'text-gray-700'}`}>{concern}</span>
                  </button>
                ))}
              </div>
              <div className="mt-10 flex flex-col items-center">
                <button onClick={handleSubmit} disabled={loading || selectedCount === 0} className={`px-8 py-3 rounded-xl font-semibold flex items-center gap-2 shadow-lg transition-all ${selectedCount > 0 ? 'bg-gradient-to-r from-yellow-600 to-yellow-600 text-white hover:from-amber-50 hover:to-amber-50 hover:shadow-xl transform hover:scale-105' : 'bg-gray-200 text-gray-400 cursor-not-allowed'}`}>
                  {loading ? <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing Your Skin...</> : <>Get Recommendations <ChevronRight className="w-5 h-5" /></>}
                </button>
                {error && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3 w-full max-w-md">
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="font-medium text-red-800">Error</h3>
                      <p className="text-red-600 text-sm">{error}</p>
                    </div>
                  </div>
                )}
                <button onClick={() => setActiveStep(1)} className="mt-6 text-rose-600 hover:text-rose-800 text-sm font-medium flex items-center gap-1 group">
                  <ChevronRight className="w-4 h-4 transform rotate-180 group-hover:-translate-x-1 transition-transform" /> Back to product selection
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Recommendations */}
        {activeStep === 3 && recommendations && (
          <div className="space-y-8">
            <div className="bg-white rounded-3xl shadow-lg overflow-hidden border border-rose-100">
              <div className="bg-gradient-to-r from-rose-50 to-pink-50 px-8 py-6 border-b border-rose-100">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Award className="text-rose-600 w-6 h-6 mr-3" />
                    <h2 className="text-2xl font-bold text-gray-800 font-serif">Your Custom Formula</h2>
                  </div>
                  <div className="flex gap-2">
                    <button onClick={() => setActiveStep(2)} className="text-rose-600 hover:text-white text-sm font-medium flex items-center gap-1 bg-white border border-gray-200 rounded-lg px-3 py-1.5 shadow-sm hover:bg-yellow-600 transition-colors">
                      Back
                    </button>
                    <button onClick={resetForm} className="text-rose-600 hover:text-white text-sm font-medium flex items-center gap-1 bg-white border border-gray-200 rounded-lg px-3 py-1.5 shadow-sm hover:bg-yellow-600 transition-colors">
                      Start Over
                    </button>
                  </div>
                </div>
              </div>

              <div className="p-6 md:p-8">
                <div className="flex items-center mb-8 bg-gradient-to-r from-yellow-100 to-yellow-100 p-4 rounded-xl border border-yellow-700">
                  <Check className="text-yellow-700 w-5 h-5 mr-3 flex-shrink-0" />
                  <p className="text-yellow-700">
                    <span className="font-semibold">Analysis Complete!</span> Based on your {selectedCount} skin concerns.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Recommended Ingredients */}
                  <div className="bg-white border border-pink-900 rounded-xl p-5 shadow-sm">
                    <h3 className="text-lg font-semibold text-pink-900 mb-4 flex items-center">
                      <div className="bg-pink-900 p-1.5 rounded-full mr-2">
                        <Check className="w-5 h-5 text-white" />
                      </div>
                      Recommended Ingredients
                    </h3>
                    
                    {groupedIngredients.recommended.length > 0 ? (
                      <div className="space-y-3">
                        {groupedIngredients.recommended.map((ingredient, index) => (
                          <div key={ingredient} className="bg-pink-900 border border-pink-900 rounded-lg p-3 flex items-start gap-3">
                            <div className="bg-gradient-to-br from-pink-900 to-pink-900 text-white w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 font-medium text-xs shadow-sm">
                              {index + 1}
                            </div>
                            <div>
                              <span className="font-medium text-white">{formatIngredientName(ingredient)}</span>
                              <p className="text-xs text-yellow-100 mt-0.5">{getIngredientDescription(ingredient)}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="bg-gray-50 rounded-lg p-4 text-gray-500 italic">
                        No specifically recommended ingredients
                      </div>
                    )}
                  </div>

                  {/* Not Recommended Ingredients */}
                  <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-700 mb-4 flex items-center">
                      <div className="bg-gray-200 p-1.5 rounded-full mr-2">
                        <X className="w-5 h-5 text-gray-600" />
                      </div>
                      Ingredients to Avoid
                    </h3>
                    
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {groupedIngredients.notRecommended.slice(0, 8).map(ingredient => (
                        <div key={ingredient} className="bg-gray-50 border border-gray-200 rounded-lg p-2 flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-gray-400"></div>
                          <span className="text-gray-600 text-sm">{formatIngredientName(ingredient)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                
                <div className="mt-8 flex justify-center">
                  <button
                    onClick={getProductRecommendations}
                    disabled={productsLoading}
                    className={`px-8 py-3 rounded-lg font-semibold flex items-center gap-2 shadow-lg transition-all ${
                      !productsLoading
                         ? 'bg-gradient-to-r from-yellow-600 to-yellow-600 text-white hover:from-amber-50 hover:to-amber-50 hover:shadow-xl transform hover:scale-105' 
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    {productsLoading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Finding Products...
                      </>
                    ) : (
                      <>
                        <ShoppingBag className="w-5 h-5" />
                        Find Matching Products
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Step 4: Product Recommendations */}
        {activeStep === 4 && recommendedProducts && (
          <div className="space-y-8">
            <div className="bg-white rounded-3xl shadow-lg overflow-hidden border border-rose-100">
              <div className="bg-gradient-to-r from-rose-50 to-pink-50 px-8 py-6 border-b border-rose-100">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <ShoppingBag className="text-rose-600 w-6 h-6 mr-3" />
                    <h2 className="text-2xl font-bold text-gray-800 font-serif">Recommended Products</h2>
                  </div>
                  <div className="flex gap-2">
                    <button onClick={() => setActiveStep(2)} className="text-rose-600 hover:text-white text-sm font-medium flex items-center gap-1 bg-white border border-gray-200 rounded-lg px-3 py-1.5 shadow-sm hover:bg-yellow-600 transition-colors">
                      Back
                    </button>
                    <button onClick={resetForm} className="text-rose-600 hover:text-white text-sm font-medium flex items-center gap-1 bg-white border border-gray-200 rounded-lg px-3 py-1.5 shadow-sm hover:bg-yellow-600 transition-colors">
                      Start Over
                    </button>
                  </div>
                </div>
              </div>

              <div className="p-6 md:p-8">
                <div className="flex items-center mb-8 bg-gradient-to-r from-yellow-100 to-yellow-100 p-4 rounded-xl border border-yellow-700">
                  <Info className="text-yellow-700 w-5 h-5 mr-3 flex-shrink-0"/>
                  <p className="text-yellow-700">
                    We found {recommendedProducts.length} products matching your needs.
                  </p>
                </div>

                {recommendedProducts.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {recommendedProducts.map((product, index) => (
                      <div key={index} className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-md hover:shadow-lg transition-shadow">
                        <div className="p-6">
                          <h3 className="font-semibold text-lg text-gray-800 mb-2">
                            {product.name || 'Unnamed Product'}
                          </h3>
                          <p className="text-gray-500 text-sm mb-4">
                            {product.brand || 'Brand'}
                          </p>
                          <p className="text-gray-500 text-sm mb-4">
                            {product.type || 'Type'}
                          </p>
                          <div>
                            <p className="text-xs font-medium text-gray-500 mb-2">Key Ingredients:</p>
                            <div className="flex flex-wrap gap-2">
                              {getProductIngredients(product, groupedIngredients.recommended).slice(0, 5).map(ing => (
                                <span key={ing} className="bg-gradient-to-r from-green-50 to-emerald-50 text-green-700 text-xs px-2 py-1 rounded-full border border-green-100 font-medium">
                                  {formatIngredientName(ing)}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="bg-gray-50 rounded-lg p-10 text-center">
                    <div className="text-gray-400 mb-3">
                      <ShoppingBag className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <h3 className="text-gray-600 font-semibold text-lg">No Products Found</h3>
                    </div>
                    <button 
                      onClick={() => setActiveStep(3)}
                      className="text-rose-600 hover:text-rose-800 text-sm font-medium flex items-center gap-1 mx-auto bg-white border border-rose-200 rounded-lg px-4 py-2 shadow-sm hover:bg-rose-50 transition-colors"
                    >
                      Back to Ingredients
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8">
        <div className="max-w-6xl mx-auto px-4 text-center">
          <p className="text-gray-400 text-sm">© {new Date().getFullYear()} GlowGuide AI. All recommendations are AI-generated.</p>
        </div>
      </footer>
    </div>
  );
}
