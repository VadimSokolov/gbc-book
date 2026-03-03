-- filters/fold-long-code.lua
-- Folds code blocks with more than THRESHOLD lines in HTML (as <details>).
-- Hides them entirely in PDF/LaTeX output.

local THRESHOLD = 5

local function count_lines(text)
  local n = 1
  for _ in text:gmatch('\n') do
    n = n + 1
  end
  return n
end

function CodeBlock(el)
  local n = count_lines(el.text)
  if n <= THRESHOLD then
    return nil  -- keep unchanged
  end

  if FORMAT:match 'html' then
    local open = pandoc.RawBlock('html',
      '<details class="code-fold">\n' ..
      '<summary>Show code (' .. tostring(n) .. ' lines)</summary>\n')
    local close = pandoc.RawBlock('html', '</details>')
    return { open, el, close }
  end

  if FORMAT:match 'latex' then
    return {}  -- omit from PDF
  end

  return nil
end
